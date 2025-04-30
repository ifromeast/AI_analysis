# 该代码只需要使用 python test_ray_tp.py 即可运行

import os
import socket
import torch
import torch.nn as nn
import torch.distributed as dist
import ray

# 构建一个从column维度切分的linear layer
class HybridParallelLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size):
        super().__init__()
        self.world_size = world_size
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
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


ray.init()

master_addr = ray._private.services.get_node_ip_address()
with socket.socket() as sock:
    sock.bind(('', 0))
    master_port = sock.getsockname()[1]

num_gpus = 4
workers = []
for i in range(num_gpus):
    options = {'runtime_env': {'env_vars': {'WORLD_SIZE': str(num_gpus), 'RANK': str(i), 'MASTER_ADDR': master_addr, 'MASTER_PORT': str(master_port)}}}
    workers.append(ray.remote(num_gpus=1)(HybridParallelLayer).options(**options).remote(128, 512, num_gpus))

batch_size = 10
input_data = torch.randn(batch_size, 128).cuda()

full_layer = torch.nn.Linear(128, 512, bias=False).cuda()
weight = full_layer.state_dict()['weight']

ret_list = []
for i in range(num_gpus):
    _ = ray.get(workers[i].load_weights.remote(weight, i))

for i in range(num_gpus):
    ret_list.append(workers[i].forward.remote(input_data))

ret = ray.get(ret_list)
ray.shutdown()

fl_ret = full_layer(input_data).cpu()
torch.testing.assert_close(ret[0], ret[1])
torch.testing.assert_close(ret[0].cpu(), fl_ret)

print("✅ 前向传播验证通过")

