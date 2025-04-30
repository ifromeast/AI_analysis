
import os


def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size