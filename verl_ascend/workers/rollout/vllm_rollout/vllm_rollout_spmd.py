from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


class vLLMRolloutPatch(vLLMRollout):
    ...


def patch_vllm_rollout_spmd():
    import verl

    from verl_ascend.patch_utils import apply_patches

    patch_list = [
        ("vLLMRollout", vLLMRolloutPatch),
    ]

    apply_patches(patch_list, verl.workers.rollout.vllm_rollout)
    apply_patches(patch_list, verl.workers.rollout.vllm_rollout.vllm_rollout_spmd)
