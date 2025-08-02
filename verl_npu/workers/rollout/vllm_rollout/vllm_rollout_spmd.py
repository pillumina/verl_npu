import gc
import os

import torch
from omegaconf import DictConfig
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

from verl_npu.patch_util import NPUPatchHelper


class vLLMRolloutPatch(NPUPatchHelper[vLLMRollout]):

    _orig_init = vLLMRollout.__init__

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        from vllm import LLM
        
        def dummy_sleep(self, level: int = 1):
            pass
        
        # disable sleep mode:
        # https://github.com/volcengine/verl/blob/53f9b2ba5e65d2e57370ed2aa59069eaa075973f/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L189
        LLM.sleep = dummy_sleep
        
        self._orig_init(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
        self.model_buffers_cpu = {}
        self.model_buffers = None
        self.worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
        for name, param in self.model.named_parameters():
            self.model_buffers_cpu[name] = torch.empty_like(param, device="cpu")

        if config.free_cache_engine:
            self.free_cache_engine()
            self.offload_model_weights()

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
        ctx = self.worker.model_runner.vllm_config.compilation_config.static_forward_context
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

        # clear kv caches
        self.worker.model_runner.kv_caches = []

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
