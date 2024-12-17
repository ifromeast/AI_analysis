import torch
from flash_attn import flash_attn_func
from reference import attention_ref
from utils import set_seed


if __name__ == "__main__":
    
    device = "cuda"
    set_seed(rank=0, seed=42)  # set seed
    batch_size = 4
    seqlen_q = seqlen_k = 1024
    d = 128
    nheads_k = nheads = 8
    dtype = torch.float16

    assert nheads % nheads_k == 0
    dropout_p = 0
    causal = True
    deterministic = False
    window_size = (-1, -1)


    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    out, lse, S_dmask = flash_attn_func(
            q, k, v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    
    out_ref, attn_ref = attention_ref(
            q, k, v,
            None,
            None,
            # attn_bias,
            # dropout_p,
            # dropout_mask,
            # causal=causal,
            # window_size=window_size,
            # softcap=softcap,
        )
    
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    (dq, dk, dv,) = torch.autograd.grad(out, (q, k, v), g)
    (dq_ref, dk_ref, dv_ref,) = torch.autograd.grad(out_ref, (q, k, v), g)
    
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")


