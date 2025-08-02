import gc

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from verl.utils import megatron_utils
from verl.utils.device import get_torch_device

from ..patch_util import NPUPatchHelper


class MegatronUtilsPatch(NPUPatchHelper[megatron_utils]):
    @torch.no_grad()
    def offload_megatron_model_to_cpu(models, offload_param=True):
        """
        In megatron, the model and optimizer storage are:
        - bf16 parameter data chunked in model parallel group
        - fp32 grad chunked in model parallel group
        - fp32 main_parameter chunked in model and dp group
        - fp32 optimizer state chunked in model and dp group
        """
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        if offload_param:
                            # offload parameters
                            if buffer.param_data.storage().size() > 0:
                                buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                                buffer.param_data_size = buffer.param_data.storage().size()
                                buffer.param_data.storage().resize_(0)

                            assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                        if buffer.grad_data.storage().size() > 0:
                            # if the grad_data size is already zero, we assume that it is already offloaded
                            buffer.grad_data_size = buffer.grad_data.storage().size()
                            buffer.grad_data.storage().resize_(0)
            else:
                # we need this for ref module
                for _, param in model_chunk.named_parameters():
                    if offload_param:
                        param.data = param.data.to("cpu", non_blocking=True)
                    if param.grad is not None:
                        param.grad = param.grad.to("cpu", non_blocking=True)
        gc.collect()
        get_torch_device().empty_cache()
