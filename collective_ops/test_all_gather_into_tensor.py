import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

shape = (1, 2)
device = torch.device("cuda:{}".format(rank))

# 每个进程的本地张量
local_tensor = torch.ones((shape[0], shape[1]), dtype=torch.int64, device=device) * rank
print(f"Rank {rank}: Local tensor = {local_tensor}")

# 创建一个用于存储所有张量的全局张量
# 注意：全局张量的大小应为 (world_size, *local_tensor.shape)
global_tensor = torch.zeros(world_size, *local_tensor.shape, dtype=local_tensor.dtype, device=device)

# 使用 all_gather_into_tensor 收集所有张量
dist.all_gather_into_tensor(global_tensor, local_tensor, group=None)

print(f"Rank {rank}: Global tensor after all_gather = {global_tensor}")


# 销毁进程组
dist.destroy_process_group()