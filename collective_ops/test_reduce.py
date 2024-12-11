import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 定义张量的形状和设备
shape = (2, 2)
device = torch.device("cuda:{}".format(rank))

# 每个进程初始化一个输入张量
input_tensor = (torch.ones(shape, dtype=torch.int64) * rank).to(device)

# 目标进程（rank 0）初始化一个输出张量
if rank == 0:
    output_tensor = torch.zeros(shape, dtype=torch.int64).to(device)
else:
    output_tensor = None

# 打印 reduce 前的张量
print(f"Rank {rank} before reduce: input_tensor={input_tensor}, output_tensor={output_tensor}")

# 执行 reduce 操作，将所有进程的 input_tensor 求和到 rank 0 的 output_tensor
dist.reduce(tensor=input_tensor, dst=0, op=dist.ReduceOp.SUM)

# 打印 reduce 后的张量
if rank == 0:
    print(f"Rank {rank} after reduce: output_tensor={input_tensor}")
else:
    print(f"Rank {rank} after reduce: input_tensor={input_tensor}")

# 销毁进程组
dist.destroy_process_group()