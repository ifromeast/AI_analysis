import torch
import torch.distributed as dist
import os

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()


# 创建一个张量并放到当前 GPU 上
tensor = torch.tensor([rank + 1.0], device=torch.device(f'cuda:{rank}'))

# 打印初始张量
print(f"Rank {rank} (GPU {rank}) has tensor {tensor}")

# 使用 all_reduce 操作
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 打印 all_reduce 后的张量
print(f"Rank {rank} (GPU {rank}) has tensor {tensor} after all_reduce")

# 销毁进程组
dist.destroy_process_group()