from verl_npu.patch_util import print_patch_summary
from verl_npu.third_party.megatron.parallel_state import ParallelStatePatch
from verl_npu.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRolloutPatch
from verl_npu.workers.megatron_workers import ActorRolloutRefWorkerPatch

def apply_npu_plugin():
    # Please make sure to apply the patches 
    # in the correct order so that they can 
    # work properly.

    # Import and apply patches only AFTER module alias injection

    ParallelStatePatch.apply_patch()
   
    vLLMRolloutPatch.apply_patch()
    
    ActorRolloutRefWorkerPatch().apply_patch()

    # Print a well-organized summary for users
    print_patch_summary()
