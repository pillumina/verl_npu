from verl_npu.utils.megatron_utils import MegatronUtilsPatch
from verl_npu.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRolloutPatch
from verl_npu.workers.sharding_manager.megatron_vllm import MegatronVLLMSshardingManagerPatch
from verl_npu.workers.megatron_workers import ActorRolloutRefWorkerPatch

def ver_npu_plugin():
    # Please make sure to apply the patches 
    # in the correct order so that they can 
    # work properly.
    MegatronUtilsPatch.apply_patch()
    vLLMRolloutPatch.apply_patch()
    MegatronVLLMSshardingManagerPatch.apply_patch()
    ActorRolloutRefWorkerPatch.apply_patch()
