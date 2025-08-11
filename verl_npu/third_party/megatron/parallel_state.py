from megatron.core import parallel_state

from verl_npu.patch_util import NPUPatchHelper

from .reloadable_pg import (
    ReloadableProcessGroup, 
    monkey_patch_torch_dist,
)


class ParallelStatePatch(NPUPatchHelper[parallel_state]):

    @staticmethod
    def destroy_process_group():
        """Destroy all reloadable process groups."""
        ReloadableProcessGroup.destroy_process_group()

    @staticmethod
    def reload_process_group():
        """Reload all reloadable process groups."""
        ReloadableProcessGroup.reload_process_groups()


monkey_patch_torch_dist()
