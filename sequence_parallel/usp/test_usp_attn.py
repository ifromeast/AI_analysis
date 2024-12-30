import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from usp.usp_attn import LongContextAttention
from usp.usp_utils import set_seq_parallel_pg, EXTRACT_FUNC_DICT
from utils import log, set_seed


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 4096
    nheads = 8
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False
    ring_attn_type = "zigzag"  # ["basic", "stripe", "zigzag"]

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

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True
    sp_ulysses_degree = 2
    sp_ring_degree = world_size // sp_ulysses_degree
    print(f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}")
    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = EXTRACT_FUNC_DICT[ring_attn_type](q, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone()
    local_k = EXTRACT_FUNC_DICT[ring_attn_type](k, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone()
    local_v = EXTRACT_FUNC_DICT[ring_attn_type](v, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone()

    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    # extract local dout
    local_dout = EXTRACT_FUNC_DICT[ring_attn_type]( dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone()

    usp_attn = LongContextAttention(ring_impl_type=ring_attn_type)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# USP forward:")
        print("#" * 30)

    out_ref, lse, _ = flash_attn_func(
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

    local_out_ref = EXTRACT_FUNC_DICT[ring_attn_type](out_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree)

    # usp attn forward
    usp_out = usp_attn(
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
    
    log("out diff", usp_out - local_out_ref)

    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB
    print(f"[Rank#{rank}] Maximum GPU memory used: {max_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device)  # Reset stats

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out_ref.backward(dout)
    local_dq_ref = EXTRACT_FUNC_DICT[ring_attn_type](q.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree)
    local_dk_ref = EXTRACT_FUNC_DICT[ring_attn_type](k.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree)
    local_dv_ref = EXTRACT_FUNC_DICT[ring_attn_type](v.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree)


    usp_out.backward(local_dout)

    log("dq diff", local_dq_ref - local_q.grad)
    log("dk diff", local_dk_ref - local_k.grad)
    log("dv diff", local_dv_ref - local_v.grad)

    if dist.is_initialized():
        dist.destroy_process_group()