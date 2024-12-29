import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from ulysses.ulysses_attn import UlyssesAttention
from utils import log, set_seed


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 8
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_k.requires_grad = True
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_v.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    sp_pg = None #dist.new_group(ranks=[i for i in range(world_size)])
    ulysses_attn = UlyssesAttention(sp_pg)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    # local_lse = lse.chunk(world_size, dim=1)[rank]

    ulysses_out = ulysses_attn(
                                local_q,
                                local_k,
                                local_v,
                                dropout_p=dropout_p,
                                causal=causal,
                                window_size=(-1, -1),
                                alibi_slopes=None,
                                deterministic=deterministic,
                                return_attn_probs=True,
                            )
    
    log("out diff", local_out - ulysses_out)
    # log("lse diff", local_lse - ulysses_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    ulysses_out.backward(local_dout)
    dist.barrier()

    out.backward(dout)
    dist.barrier()
    dq, dk, dv = q.grad, k.grad, v.grad
    local_dq_ref = dq.chunk(world_size, dim=1)[rank]
    local_dk_ref = dk.chunk(world_size, dim=1)[rank]
    local_dv_ref = dv.chunk(world_size, dim=1)[rank]

    log("dq diff", local_dq_ref - local_q.grad)
    log("dk diff", local_dk_ref - local_k.grad)
    log("dv diff", local_dv_ref - local_v.grad)

