import torch
import torch.distributed as dist
import os

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda:{}".format(rank))

# 创建一个简单的张量
tensor = torch.tensor([rank, rank+1], dtype=torch.float32, device=device)
print("rank {} tensor {}".format(rank, tensor))

send_op = dist.P2POp(dist.isend, tensor, (rank + 1)%world_size)

recv_tensor = torch.zeros((1,2), dtype=torch.float32, device=device)
recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
reqs = dist.batch_isend_irecv([send_op, recv_op])

for req in reqs:
    req.wait()
print(f"Rank {rank} received: {recv_tensor}")


# 销毁进程组
dist.destroy_process_group()


    
  

