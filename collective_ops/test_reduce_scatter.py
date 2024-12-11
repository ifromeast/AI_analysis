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
local_tensor = (torch.ones((shape[0], shape[1]), dtype=torch.int64) * rank).to(device)

# 用于存储 reduce_scatter 结果的张量
output_tensor = torch.zeros((shape[0] // world_size, shape[1]), dtype=torch.int64).to(device)

# 打印 reduce_scatter 前的张量
print(f"Rank {rank} before reduce_scatter: local_tensor={local_tensor}, output_tensor={output_tensor}")

# 执行 reduce_scatter 操作
dist.reduce_scatter(output_tensor, [local_tensor for _ in range(world_size)])

# 打印 reduce_scatter 后的张量
print(f"Rank {rank} after reduce_scatter: output_tensor={output_tensor}")

# 销毁进程组
dist.destroy_process_group()