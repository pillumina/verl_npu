from verl_ascend.models.mcore.util import patch_mcore_util
from verl_ascend.workers.rollout.vllm_rollout.vllm_rollout_spmd import patch_vllm_rollout_spmd

def adpat_verl_to_ascend():
    from mindspeed import megatron_adaptor  # noqa: F401

    patch_mcore_util()
    patch_vllm_rollout_spmd()
