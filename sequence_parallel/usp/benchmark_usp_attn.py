
import os
import torch
import torch.distributed as dist
import argparse
from utils import flops, efficiency
from usp.usp_attn import LongContextAttention
from usp.usp_utils import set_seq_parallel_pg, EXTRACT_FUNC_DICT


def benchmark(num_iter=100, forward_only=True, log=True, profile=False):
    dtype = torch.float16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    seqlen = args.seq_len
    nheads = args.nheads
    d = args.head_size

    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0, f"seqlen {seqlen} world_size {world_size}"
    assert d % 8 == 0
    # prepare process group for hybrid sequence parallelism
    assert world_size == args.sp_ulysses_degree * args.sp_ring_degree
    if rank == 0:
        print(f"rank {rank}, sp_ulysses_degree: {args.sp_ulysses_degree}, sp_ring_degree: {args.sp_ring_degree}")

    set_seq_parallel_pg(args.sp_ulysses_degree, args.sp_ring_degree, rank, world_size)

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = EXTRACT_FUNC_DICT[args.ring_attn_type](q, rank, world_size=world_size, rd=args.sp_ring_degree, ud=args.sp_ulysses_degree).detach().clone()
    local_k = EXTRACT_FUNC_DICT[args.ring_attn_type](k, rank, world_size=world_size, rd=args.sp_ring_degree, ud=args.sp_ulysses_degree).detach().clone()
    local_v = EXTRACT_FUNC_DICT[args.ring_attn_type](v, rank, world_size=world_size, rd=args.sp_ring_degree, ud=args.sp_ulysses_degree).detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    # extract local dout
    local_dout = EXTRACT_FUNC_DICT[args.ring_attn_type]( dout, rank, world_size=world_size, rd=args.sp_ring_degree, ud=args.sp_ulysses_degree).detach().clone()

    usp_attn = LongContextAttention(ring_impl_type=args.ring_attn_type)

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5,),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./benchmark/profiles/usp_bs_{batch_size}_seq_{seqlen}_heads_{nheads}_d_{d}_rank_{dist.get_rank()}_fwd_only_{forward_only}_ulysses_{args.sp_ulysses_degree}_ring_{args.sp_ring_degree}_ring_attn_{args.ring_attn_type}",
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    # warmup
    out = usp_attn(
                    local_q,
                    local_k,
                    local_v,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )
    out.backward(local_dout)

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = usp_attn(
                            local_q,
                            local_k,
                            local_v,
                            dropout_p=dropout_p,
                            causal=causal,
                            window_size=(-1, -1),
                            alibi_slopes=None,
                            deterministic=deterministic,
                            return_attn_probs=False,
                        )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            local_q.grad = None
            local_k.grad = None
            local_v.grad = None
            out = usp_attn(
                            local_q,
                            local_k,
                            local_v,
                            dropout_p=dropout_p,
                            causal=causal,
                            window_size=(-1, -1),
                            alibi_slopes=None,
                            deterministic=deterministic,
                            return_attn_probs=False,
                        )
            out.backward(local_dout)
            if profile:
                profiler.step()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end)/1000

    if profile:
        profiler.stop()

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iters/s, {time*1000/num_iter:.3f} ms/iter")
        print(f"peak memory: {torch.cuda.max_memory_allocated(device=device) / 1024 / 1024:.3f} MB")
        torch.cuda.reset_peak_memory_stats(device)   # Reset stats
        mode = "fwd" if forward_only else "fwd_bwd"
        speed_f = efficiency(
                    flops(batch_size, seqlen, d, nheads, causal, mode=mode),
                    time/num_iter
                )
        print(f"speed: {speed_f:.3f} TFLOPs/s")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--nheads", type=int, default=16, help="head number")
    parser.add_argument("--head_size", type=int, default=128, help="head dimension")
    parser.add_argument("--seq_len", type=int, default=4 * 1024, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--fwd_only", action="store_true", help="benchmark forward pass only")
    parser.add_argument("--profile", action="store_true", help="generate torch profile or not")
    parser.add_argument('--sp_ulysses_degree', type=int, default=2, help='sp_ulysses_degree (default: world_size)')
    parser.add_argument('--sp_ring_degree', type=int, default=4, help='sp_ring_degree (default: world_size)')
    parser.add_argument('--ring_attn_type', type=str, default='basic', choices=['basic', 'stripe', 'zigzag'], help='ring attention type (default: basic)')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    if rank == 0:
        print(f"usp BS:{args.batch_size} seq_len:{args.seq_len} nheads:{args.nheads} head_size:{args.head_size}, fwd_only: {args.fwd_only} ulysses_degree: {args.sp_ulysses_degree} ring_degree: {args.sp_ring_degree} ring_attn_type: {args.ring_attn_type}")
    benchmark(num_iter=100, forward_only=args.fwd_only, log=True, profile=args.profile)

