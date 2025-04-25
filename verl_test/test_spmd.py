
import os
import torch
import torch.nn as nn

torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ['RANK']))

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 构建一个从column维度切分的linear layer
class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size // int(os.environ['WORLD_SIZE']), bias=False).to(device='cuda')

    def forward(self, x):
        ret = self.layer(x.to(device='cuda'))
        output_tensor = torch.zeros(size=(int(os.environ['WORLD_SIZE']), ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)
        torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)
        output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)

        return output_tensor

    def load_weights(self, weight):
        rank = int(os.environ['RANK'])

        world_size = int(os.environ['WORLD_SIZE'])
        dim_per_rank = weight.shape[0] // world_size
        self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])

batch_size = 10
input_data = torch.randn(batch_size, 128)

# init一个PyTorch的linear layer，并让我们构建的layer和它保持参数一致。
full_layer = torch.nn.Linear(128, 512, bias=False)
weight = full_layer.state_dict()['weight']

tp_layer = ColumnTPLayer(128, 512)
tp_layer.load_weights(weight)

tp_ret = tp_layer(input_data).cpu()
fl_ret = full_layer(input_data).cpu()

torch.testing.assert_close(tp_ret, fl_ret)
