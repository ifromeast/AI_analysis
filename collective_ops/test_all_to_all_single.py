import torch
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

device = torch.device("cuda:{}".format(rank))

# 输入张量
if rank == 0:
    input = torch.tensor([0, 1, 2, 3, 4, 5]).to(device)
    input_splits = [2, 2, 1, 1]
    output_splits = [2, 3, 2, 2] 
elif rank == 1:
    input = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18]).to(device)
    input_splits = [3, 2, 2, 2]
    output_splits = [2, 2, 1, 2]
elif rank == 2:
    input = torch.tensor([20, 21, 22, 23, 24]).to(device)
    input_splits = [2, 1, 1, 1]
    output_splits = [1, 2, 1, 2]
elif rank == 3:
    input = torch.tensor([30, 31, 32, 33, 34, 35, 36]).to(device)
    input_splits = [2, 2, 2, 1]
    output_splits = [1, 2, 1, 1]


output = torch.empty(sum(output_splits), dtype=torch.int64).to(device)

# 调用 all_to_all_single
dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits)

print(f"Rank {rank}: Input = {input}, Output = {output}")

# 销毁进程组
dist.destroy_process_group()
