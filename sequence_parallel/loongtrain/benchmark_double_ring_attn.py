import os
import torch
import torch.distributed as dist
from ring_flash_attention.zigzag_ring_flash_attn import extract_local
from loongtrain.double_ring_attn import zigzag_ring_flash_attn_func_with_sliding_window
from loongtrain.double_ring_utils import generate_2d_attn_process_group
import argparse
from utils import flops, efficiency


def benchmark(f, num_iter=100, forward_only=True, log=True, profile=False):
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

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./benchmark/profiles/{f.__name__}_bs_{batch_size}_seq_{seqlen}_heads_{nheads}_d_{d}_rank_{dist.get_rank()}_fwd_only_{forward_only}"
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    # warmup
    out, lse, _ = f(
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
    out.backward(local_dout)

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
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
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            local_q.grad = None
            local_k.grad = None
            local_v.grad = None
            out, lse, _ = f(
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
            out.backward(local_dout)
            if profile:
                profiler.step()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000

    if profile:
        profiler.stop()

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iters/s, {time*1000/num_iter:.3f} ms/iter")
        print(
            f"peak memory: {torch.cuda.max_memory_allocated(device=device) / 1024 / 1024:.3f} MB"
        )
        mode = "fwd" if forward_only else "fwd_bwd"
        speed_f = efficiency(
            flops(batch_size, seqlen, d, nheads, causal, mode=mode), time / num_iter
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
    parser.add_argument(
        "--fwd_only", action="store_true", help="benchmark forward pass only"
    )
    parser.add_argument(
        "--profile", action="store_true", help="generate torch profile or not"
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()
    if rank == 0:
        print(
            f"{zigzag_ring_flash_attn_func_with_sliding_window.__name__} BS:{args.batch_size} seq_len:{args.seq_len} nheads:{args.nheads} head_size:{args.head_size}, fwd_only: {args.fwd_only}"
        )
    benchmark(
        zigzag_ring_flash_attn_func_with_sliding_window,
        num_iter=100,
        forward_only=args.fwd_only,
        log=True,
        profile=args.profile,
    )
