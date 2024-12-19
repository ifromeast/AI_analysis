import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

shape = (16,2)
device = torch.device("cuda:{}".format(rank))

output_tensor=torch.zeros((shape[0]//world_size,shape[1]),dtype=torch.int64).to(device)

# 在 root 进程中创建一个列表来发送所有张量
if rank == 0:
    tensor_list=[(torch.ones((shape[0]//world_size,shape[1]),dtype=torch.int64)*i).to(device) for i in range(world_size)]

# # 打印 scatter 前的张量
if rank == 0:
    print(f"Rank {rank} before scatter: scattered_tensors={tensor_list}, recv_tensor={output_tensor}")
else:
    print(f"Rank {rank} before scatter: recv_tensor={output_tensor}")

if rank == 0:
    dist.scatter(output_tensor, scatter_list=tensor_list, src=0)
else:
    dist.scatter(output_tensor, src=0)

# # 打印 scatter 后的张量
print(f"Rank {rank} after scatter: recv_tensor={output_tensor}")

# 销毁进程组
dist.destroy_process_group()