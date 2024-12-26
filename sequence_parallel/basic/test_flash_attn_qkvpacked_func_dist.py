import random
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from utils import set_seed, log
from reference import attention_ref


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(qkv, src=0)

    q, k, v = qkv.clone().unbind(2)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank].detach().clone()
    local_lse = lse.chunk(world_size, dim=-1)[rank]


    out_pt_ref, attn_pt_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        None,
        dropout_p,
        dropout_mask=None,
        causal=causal,
        window_size=(-1, -1),
    )
    local_out_pt_ref = out_pt_ref.chunk(world_size, dim=1)[rank].detach().clone()
    dist.barrier()

    print(f'rank {rank} out (distributed) - out_ref (non-distributed) diff: {(local_out - local_out_pt_ref).abs().max().item()}')

    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dqkv = qkv.grad
    local_dqkv = dqkv.chunk(world_size, dim=1)[rank].detach().clone()

    (dq_ref, dk_ref, dv_ref,) = torch.autograd.grad(out_pt_ref, (q, k, v), dout)
    local_dq_ref = dq_ref.chunk(world_size, dim=1)[rank].detach().clone()
    local_dk_ref = dk_ref.chunk(world_size, dim=1)[rank].detach().clone()
    local_dv_ref = dv_ref.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()

    log("dq diff", local_dqkv[:, :, 0] - local_dq_ref)
    log("dk diff", local_dqkv[:, :, 1] - local_dk_ref)
    log("dv diff", local_dqkv[:, :, 2] - local_dv_ref)

    if dist.is_initialized():
        dist.destroy_process_group()
