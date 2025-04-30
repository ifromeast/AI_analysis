
import ray
import torch
import torch.nn as nn
import socket
import os
import logging
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Follow defination above
@ray.remote
class DPModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.layer = nn.Linear(input_size, output_size, bias=False).cuda()

    def forward(self, x):
        x = dp_data_future_process(x)
        return self.layer(x.cuda())

    def load_weights(self, weight):
        self.layer.weight.data.copy_(weight)

    def state_dict(self):
        return self.layer.state_dict()

@ray.remote
class ColumnTPModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.layer = nn.Linear(input_size, output_size // int(os.environ['WORLD_SIZE']), bias=False).cuda()

    def forward(self, x):
        if isinstance(x, ray.ObjectRef):
            x = ray.get(x)
        elif isinstance(x, list) and isinstance(x[0], ray.ObjectRef):
            x = torch.cat(ray.get(x), dim=0)

        ret = self.layer(x.cuda())

        output_tensor = torch.zeros(size=(int(os.environ['WORLD_SIZE']), ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)

        torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)

        output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)

        return output_tensor

    def load_weights(self, weight):
        rank = int(os.environ['RANK'])

        world_size = int(os.environ['WORLD_SIZE'])
        dim_per_rank = weight.shape[0] // world_size
        self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])

    def state_dict(self):
        return self.layer.state_dict()

def create_placement_group(num_gpus):
    """Create and return a placement group for GPU allocation."""
    bundles = [{"CPU": 4, "GPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles=bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg

def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(('', 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port

def create_worker_options(pg, rank, num_gpus, master_addr, master_port):
    """Create options for Ray workers."""
    return {
        'runtime_env': {
            'env_vars': {
                'WORLD_SIZE': str(num_gpus),
                'RANK': str(rank),
                'MASTER_ADDR': master_addr,
                'MASTER_PORT': str(master_port)
            }
        },
        'scheduling_strategy': PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=rank
        ),
        'num_gpus': 0.2
    }

def dp_data_future_process(x):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if isinstance(x, list):
        if isinstance(x[0], ray.ObjectRef):
            x = torch.cat(ray.get(x), dim=0)
        else:
            x = torch.cat(x, dim=0)

    if isinstance(x, torch.Tensor):
        bs_per_rank = x.shape[0] // world_size
        x = x[bs_per_rank*rank: bs_per_rank*(rank+1)]

    elif isinstance(x, ray.ObjectRef):
        x = ray.get(x)
    return x

# Initialize Ray
ray.init()

# Create placement group
num_gpus = 2
pg = create_placement_group(num_gpus)

# Layer configurations
layer_configs = [
    {'type': DPModel, 'num_gpus': 1, 'input_size': 32, 'output_size': 64},
    {'type': ColumnTPModel, 'num_gpus': 2, 'input_size': 64, 'output_size': 128},
    {'type': DPModel, 'num_gpus': 2, 'input_size': 128, 'output_size': 4}
]

# Create all layer workers
all_worker_group = []
for config in layer_configs:
    master_addr, master_port = get_network_config()
    worker_group = []
    for i in range(config['num_gpus']):
        options = create_worker_options(pg, i, config['num_gpus'], master_addr, master_port)
        worker_group.append(config['type'].options(**options).remote(config['input_size'], config['output_size']))
    all_worker_group.append(worker_group)

worker_group_1, worker_group_2, worker_group_3 = all_worker_group 

# Prepare input data and reference layers
batch_size = 10
input_data = torch.randn(batch_size, 32)

reference_layers = [
    torch.nn.Linear(32, 64, bias=False),
    torch.nn.Linear(64, 128, bias=False),
    torch.nn.Linear(128, 4, bias=False)
]
weights = [layer.state_dict()['weight'] for layer in reference_layers]

# Load weights
ray.get(worker_group_1[0].load_weights.remote(weights[0]))
for i in range(2):  # max_gpus
    ray.get(worker_group_2[i].load_weights.remote(weights[1]))
    ray.get(worker_group_3[i].load_weights.remote(weights[2]))

cur_data = ray.put(input_data)

# Forward passes
# First layer (DP with 1 GPU)
batch_per_gpus = batch_size // num_gpus
cur_result = worker_group_1[0].forward.remote(cur_data)
ret = [cur_result]

# Second layer (ColumnTP with 2 GPUs)
ret_list_2 = []
batch_per_gpus = batch_size // num_gpus
for i in range(num_gpus):
    cur_result = worker_group_2[i].forward.remote(ret)
ret_2 = [cur_result]

# Third layer (DP with 2 GPUs)
final_list = []
for i in range(num_gpus):
    cur_data = ret_2
    cur_result = worker_group_3[i].forward.remote(cur_data)
    final_list.append(cur_result)

# Get and combine results
final = ray.get(final_list)
ret = torch.cat(final, dim=0)

# Compare with reference implementation
fl_ret = input_data
for i in range(3):
    fl_ret = reference_layers[i](fl_ret)

torch.testing.assert_close(ret.cpu(), fl_ret)
print("✅ 前向传播验证通过")

# Cleanup
ray.shutdown()
