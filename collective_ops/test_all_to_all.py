import torch
import torch.distributed as dist

# 初始化进程组
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

device = torch.device("cuda:{}".format(rank))

input = torch.arange(8) + rank * 8
input = list(input.to(device).chunk(8))
print(f"Rank {rank} before all_to_all input={input}")

output = list(torch.empty([8], dtype=torch.int64).to(device).chunk(8))
dist.all_to_all(output, input)
print(f"Rank {rank} after all_to_all output={output}")

# 销毁进程组
dist.destroy_process_group()
