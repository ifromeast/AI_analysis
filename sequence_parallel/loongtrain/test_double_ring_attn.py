import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from loongtrain.double_ring_attn import zigzag_ring_flash_attn_func_with_sliding_window
from ring_flash_attention.zigzag_ring_flash_attn import extract_local
from loongtrain.double_ring_utils import generate_2d_attn_process_group
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

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    # prepare process group for double ring attention sequence parallelism
    context_parallel_size = 8
    double_ring_window_size = 4

    group_results = generate_2d_attn_process_group(
        world_size,
        rank,
        head_size=1,
        context_size=context_parallel_size,
        window_size=double_ring_window_size,
        head_first=True,
        interleaved=False,
        sp_size=world_size,
        with_cpu_group=False,
    )

    for item in group_results:
        if item[5] == "head":
            head_group = item[2]
        elif item[5] == "context":
            context_group = item[2]
        elif item[5] == "intra_window":
            intra_window_group = item[2]
        elif item[5] == "inter_window":
            inter_window_group = item[2]
        elif item[5] == "dkv_intra_window":
            dkv_intra_window_group = item[2]
        elif item[5] == "dkv_inter_window":
            dkv_inter_window_group = item[2]

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = extract_local(q, rank, world_size).detach().clone()
    local_k = extract_local(k, rank, world_size).detach().clone()
    local_v = extract_local(v, rank, world_size).detach().clone()

    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    # extract local dout
    local_dout = extract_local(dout, rank, world_size).detach().clone()

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

    local_out_ref = extract_local(out_ref, rank, world_size).detach().clone()

    # usp attn forward
    double_ring_out, double_ring_lse, _ = (
        zigzag_ring_flash_attn_func_with_sliding_window(
            local_q,
            local_k,
            local_v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
            context_group=context_group,
            inter_window_group=inter_window_group,
            intra_window_group=intra_window_group,
            dkv_inter_window_group=dkv_inter_window_group,
            dkv_intra_window_group=dkv_intra_window_group,
            double_ring_window_size=double_ring_window_size,
        )
    )

    log("out diff", double_ring_out - local_out_ref)

    max_memory = torch.cuda.max_memory_allocated(device) / (
        1024 * 1024
    )  # Convert to MB
    print(f"[Rank#{rank}] Maximum GPU memory used: {max_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device)  # Reset stats

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out_ref.backward(dout)
    local_dq_ref = extract_local(q.grad, rank, world_size)
    local_dk_ref = extract_local(k.grad, rank, world_size)
    local_dv_ref = extract_local(v.grad, rank, world_size)

    double_ring_out.backward(local_dout)

    log("dq diff", local_dq_ref - local_q.grad)
    log("dk diff", local_dk_ref - local_k.grad)
    log("dv diff", local_dv_ref - local_v.grad)

    if dist.is_initialized():
        dist.destroy_process_group()
