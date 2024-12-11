import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

shape = (8, 2)
device = torch.device("cuda:{}".format(rank))

# 每个进程的本地张量
local_tensor = (torch.ones((shape[0] // world_size, shape[1]), dtype=torch.int64) * rank).to(device)

# 用于存储所有进程张量的列表
output_tensor_list = [torch.zeros((shape[0] // world_size, shape[1]), dtype=torch.int64).to(device) for _ in range(world_size)]

# 打印 all_gather 前的张量
print(f"Rank {rank} before all_gather: local_tensor={local_tensor}, output_tensor_list={output_tensor_list}")

# 执行 all_gather 操作
dist.all_gather(output_tensor_list, local_tensor)

# 打印 all_gather 后的张量
print(f"Rank {rank} after all_gather: output_tensor_list={output_tensor_list}")

# 销毁进程组
dist.destroy_process_group()