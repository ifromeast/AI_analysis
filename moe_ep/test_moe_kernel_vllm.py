import torch

import torch.nn.functional as F

from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)

import pdb

@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.silu_and_mul

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_neuron(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x_reshaped = x.view(-1, x.shape[-1])
        s = x_reshaped[:, :d] * F.sigmoid(x_reshaped[:, :d])
        result = s * x_reshaped[:, d:]
        return result.view(*x.shape[:-1], d)


def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    
    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    triton_output = fused_moe(a,
                              w1,
                              w2,
                              score,
                              topk,
                              global_num_experts=e,
                              expert_map=e_map,
                              renormalize=False)
    
    torch_output = torch_moe(a, w1, w2, score, topk, e_map)
    torch.allclose(triton_output, torch_output, atol=2e-2, rtol=0)
    iterative_output = iterative_moe(a,
                                     w1,
                                     w2,
                                     score,
                                     topk,
                                     global_num_experts=e,
                                     expert_map=e_map,
                                     renormalize=False)
    torch.allclose(iterative_output,
                               torch_output,
                               atol=2e-2,
                               rtol=0)

if __name__ == "__main__":
    NUM_EXPERTS = 8
    EP_SIZE = 4
    TOP_KS = 3

    m = 33
    n = 128
    k = 512
    e = NUM_EXPERTS
    topk = TOP_KS
    ep_size = EP_SIZE
    dtype = torch.float16
    test_fused_moe(m, n, k, e, topk, ep_size, dtype)

