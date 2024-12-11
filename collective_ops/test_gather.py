import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

shape = (16, 2)
device = torch.device("cuda:{}".format(rank))

# 每个进程生成一个张量
input_tensor = (torch.ones((shape[0] // world_size, shape[1]), dtype=torch.int64) * rank).to(device)

# 在 root 进程中创建一个列表来接收所有张量
if rank == 1:
    tensor_list = [torch.zeros((shape[0] // world_size, shape[1]), dtype=torch.int64).to(device) for _ in range(world_size)]
else:
    tensor_list = None

# 打印 gather 前的张量
print(f"Rank {rank} before gather: input_tensor={input_tensor}")

# 执行 gather 操作
dist.gather(input_tensor, gather_list=tensor_list, dst=1)

# 打印 gather 后的张量
if rank == 1:
    print(f"Rank {rank} after gather: gathered_tensors={tensor_list}")
else:
    print(f"Rank {rank} after gather: input_tensor={input_tensor}")

# 销毁进程组
dist.destroy_process_group()