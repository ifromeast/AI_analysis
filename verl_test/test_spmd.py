# 该代码需要使用 torchrun --nproc_per_node=4 test_spmd.py 来运行

import os
import torch
import torch.nn as nn
import torch.distributed as dist

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ['RANK']))
world_size = dist.get_world_size()
rank = dist.get_rank()
print(f"Rank {rank} of {world_size} initializing...")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 构建一个从column维度切分的linear layer
class HybridParallelLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size):
        super().__init__()
        self.world_size = world_size
        self.layer = nn.Linear(input_size, output_size // world_size, bias=False).to(device='cuda')

    def forward(self, x):
        local_output = self.layer(x)
        
        # 跨设备收集所有分片结果
        output_list = [torch.empty_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        # 沿特征维度拼接
        return torch.cat(output_list, dim=-1)

    def load_weights(self, weight, rank):
        dim_per_rank = weight.shape[0] // self.world_size
        self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])


# 数据准备 -----------------------------------------------------------------
batch_size_per_gpu = 16
global_batch_size = batch_size_per_gpu * world_size

# 生成全局数据（模拟数据加载器行为）
global_input = torch.randn(global_batch_size, 128).cuda()  # 由于 seed 固定，每个 rank 数据是一样的
local_input = global_input.chunk(world_size, dim=0)[rank].detach().clone()

full_layer = torch.nn.Linear(128, 512, bias=False).cuda()
weight = full_layer.weight.data
print(f"full_layer weight shape: {weight.shape}")

tp_layer = HybridParallelLayer(128, 512, world_size)
tp_layer.load_weights(weight, rank)

tp_ret = tp_layer(global_input)   # TP 输入的数据必须一样
with torch.no_grad():
    fl_ret = full_layer(local_input)

torch.testing.assert_close(tp_ret.chunk(world_size, dim=0)[rank].detach().cpu(), 
                           fl_ret.detach().cpu(), atol=1e-3, rtol=1e-3)
if rank == 0:
    print("✅ 前向传播验证通过")

