import gc
import os
from copy import deepcopy

import ray
import torch
import vllm.envs as envs
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

from verl_npu.patch_util import NPUPatchHelper


class vLLMRolloutPatch(NPUPatchHelper[vLLMRollout]):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        # patch this for npu
        enable_infer_ep = False
        if hasattr(config, "dp_model_parallel_size") and config.dp_model_parallel_size > 1:
            self._init_dp_envs(config)
            enable_infer_ep = True

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            enable_expert_parallel=enable_infer_ep,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
        self.model_buffers_cpu = {}
        self.model_buffers = None
        self.worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
        for name, param in self.model.named_parameters():
            self.model_buffers_cpu[name] = torch.empty_like(param, device="cpu")

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.free_cache_engine()
            self.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    def _init_dp_env(self, config):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # world_size = int(config.get("rollout_world_size", 1))
        tp_size = int(config.get("tensor_model_parallel_size", 1))
        dp_size = int(config.get("dp_model_parallel_size", 1))

        all_ranks = torch.arange(world_size).reshape(-1, dp_size, 1, tp_size)  # noqa
        group_ranks = all_ranks.transpose(1, 3).reshape(-1, dp_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # all gather ip
        ip_addr = ray.util.get_node_ip_address()
        ip_list = [None] * world_size
        torch.distributed.all_gather_object(ip_list, ip_addr)

        for index, group_rank in enumerate(group_ranks):
            if torch.distributed.get_rank() in group_rank:
                os.environ["VLLM_DP_MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1 + index)
                os.environ["VLLM_DP_MASTER_IP"] = ip_list[group_rank[0]]
        local_dp_rank = rank // tp_size % dp_size
        os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_PORT"] = os.environ["VLLM_DP_MASTER_PORT"]
        envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])
        envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
        envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])

        print(f"[VLLM] using TP={tp_size}, DP={dp_size}", flush=True)

    def onload_model_weights(self):
        self.model_buffers = {}
        for name, param in self.model.named_parameters():
            self.model_buffers[name] = torch.empty_like(param, device=get_device_name())
            param.data = self.model_buffers[name]

    def offload_model_weights(self):
        for name, params in self.model.named_parameters():
            params.data = self.model_buffers_cpu[name]

        self.model_buffers = None
        gc.collect()
        get_torch_device().empty_cache()

    def init_cache_engine(self):
        if os.environ["VLLM_USE_V1"] == "1":
            if not self.worker.model_runner.kv_caches:
                self.inference_engine.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.inference_engine.llm_engine.engine_core.engine_core.kv_cache_configs)
                self.inference_engine.llm_engine.reset_prefix_cache()
        else:
            if self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.inference_engine.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def free_cache_engine(self):
        if os.environ["VLLM_USE_V1"] == "1":
            ctx = self.worker.model_runner.vllm_config.compilation_config.static_forward_context
        else:
            ctx = self.inference_engine.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = self.inference_engine.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache
        if os.environ["VLLM_USE_V1"] == "1":
            self.worker.model_runner.kv_caches = []
        else:
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine = None
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.gpu_cache = None
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        # 多模态kv cache
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.language_model.model.start_layer, self.model.language_model.model.end_layer):
                attn_impl = self.model.language_model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
