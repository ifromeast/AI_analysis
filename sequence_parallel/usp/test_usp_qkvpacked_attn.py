import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from usp.usp_attn import LongContextAttentionQKVPacked
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
    ring_attn_type = "basic"  # ["basic", "stripe", "zigzag"]

    assert seqlen % world_size == 0
    assert d % 8 == 0

    # global tensors
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    with torch.no_grad():
        dist.broadcast(qkv, src=0)
        dist.broadcast(dout, src=0)

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True
    sp_ulysses_degree = 2
    sp_ring_degree = world_size // sp_ulysses_degree
    print(f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}")
    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # sharded tensors for long context attn
    local_qkv = (EXTRACT_FUNC_DICT[ring_attn_type](qkv, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone())
    local_qkv.requires_grad = True

    local_dout = (EXTRACT_FUNC_DICT[ring_attn_type](dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree).detach().clone())

    usp_attn = LongContextAttentionQKVPacked(ring_impl_type=ring_attn_type)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# USP forward:")
        print("#" * 30)

    out_ref, lse, _ = flash_attn_qkvpacked_func(
                                                qkv,
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
                        local_qkv,
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
    dqkv = qkv.grad
    local_dqkv_ref = EXTRACT_FUNC_DICT[ring_attn_type](dqkv, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree)

    usp_out.backward(local_dout)
    local_dqkv = local_qkv.grad

    log("dq diff", local_dqkv_ref[:,:,0] - local_dqkv[:,:,0])
    log("dk diff", local_dqkv_ref[:,:,1] - local_dqkv[:,:,1])
    log("dv diff", local_dqkv_ref[:,:,2] - local_dqkv[:,:,2])

    if dist.is_initialized():
        dist.destroy_process_group()