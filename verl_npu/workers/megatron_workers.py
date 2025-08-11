import psutil
import warnings
from codetiming import Timer
from megatron.core import mpu
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.megatron_workers import ActorRolloutRefWorker, logger
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.profiler import (
    DistProfiler, 
    GPUMemoryLogger,
    log_gpu_memory_usage,
)
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl_npu.patch_util import NPUPatchHelper


class ActorRolloutRefWorkerPatch(NPUPatchHelper[ActorRolloutRefWorker]):

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        # reload hccl groups
        if hasattr(mpu, "reload_process_groups"):
            mpu.reload_process_groups()

        data.batch = data.batch.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        # offload hccl groups
        if hasattr(mpu, "destroy_process_groups"):
            mpu.destroy_process_groups()

        aggressive_empty_cache(force_sync=True)
        return output


    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        if self.config.rollout.name == "vllm":

            from torch.distributed.device_mesh import init_device_mesh

            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager
            from verl.workers.sharding_manager.hybrid_tp_config import HybridTPConfig

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))

            # create HybridTPConfig
            hybrid_tp_config = HybridTPConfig.from_dict_config(
                self.config.rollout.get("hybrid_tp", {}),
                self.config.rollout.tensor_model_parallel_size
            )

            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    hybrid_tp_config=hybrid_tp_config,
                )
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    hybrid_tp_config=hybrid_tp_config,
                )

            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            # perform weight resharding between actor and rollout
            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

            
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                bridge=None,
                hybrid_tp_config=hybrid_tp_config,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif self.config.rollout.name in ["sglang", "sglang_async"]:
            if self.config.rollout.name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
            # However, due to verl's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.megatron_sglang import MegatronSGLangShardingManager

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)

            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronSGLangShardingManager(
                actor_module=self.actor.actor_module,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError("Only vllmRollout is supported with Megatron now")
        print(f"rollout and sharding manager init done sharding_manager: {sharding_manager}")
        return rollout, sharding_manager