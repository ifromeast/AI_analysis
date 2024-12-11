import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

tensor = torch.tensor([rank + 1.0, rank - 1.0], device=torch.device(f'cuda:{rank}'))

# 打印广播前的张量
print(f"Rank {rank} before broadcast: {tensor}")

# 使用 broadcast 操作将 rank 0 的张量广播到所有进程
dist.broadcast(tensor, src=0)

# 打印广播后的张量
print(f"Rank {rank} after broadcast: {tensor}")

# 销毁进程组
dist.destroy_process_group()