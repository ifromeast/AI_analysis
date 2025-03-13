import os
import math
import json
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from deepseek.configuration_deepseek import DeepseekV2Config 
from deepseek.modeling_deepseek import DeepseekV2MoE



def init_parallel_groups(ep_size=1):
    dist.init_process_group("nccl")
    # world_size = int(os.getenv("WORLD_SIZE", "0"))
    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    ep_group = edp_group = None
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            ep_group = group
    edp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            edp_group = group
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    dist.all_reduce(torch.zeros(1, device="cuda"), group=edp_group)
    return world_size, local_rank, ep_group, edp_group


if __name__ == "__main__":
    ep_size = 8
    batch_size = 8
    seq_length = 4096
    only_forward = True

    world_size, local_rank, ep_group, edp_group = init_parallel_groups(ep_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    with open("deepseek/config.json", "r") as f:
        config_dict = json.load(f)

    config = DeepseekV2Config(**config_dict)
    config.ep_size = ep_size
    dsv2_moe = DeepseekV2MoE(config).to(device)

    input_x = torch.randn(batch_size, seq_length, config.hidden_size).to(device)
    dout = torch.randn(batch_size, seq_length, config.hidden_size).to(device)

    moe_outputs = dsv2_moe(input_x) # warmup
    moe_outputs.backward(dout)

    torch.backends.cudnn.benchmark = True
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1,),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./profiles/dsv2moe_ep_bs_{batch_size}_seq_{seq_length}_rank_{local_rank}_{'fwd' if only_forward else 'fwd_bwd'}"
        ),
    )
    profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    for i in range(8):
        if only_forward:
            with torch.no_grad():
                moe_outputs = dsv2_moe(input_x)
        else:
            moe_outputs = dsv2_moe(input_x)
            moe_outputs.backward(dout)
        profiler.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    profiler.stop()
    dist.destroy_process_group()
