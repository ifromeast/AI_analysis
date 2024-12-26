import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_qkvpacked_func
from reference import attention_ref


if __name__ == "__main__":
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    seqlen = 1024
    d = 128
    nheads_k = nheads = 8
    dtype = torch.float16
    dropout_p = 0
    causal = True
    deterministic = False
    window_size=(-1, -1)
    alibi_slopes, attn_bias = None, None
    dropout_mask = None

    assert nheads % nheads_k == 0
    # window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    q, k, v = qkv.clone().unbind(2)

    out, lse, S_dmask = flash_attn_qkvpacked_func(
            qkv,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    
    out_ref, attn_ref = attention_ref(
            q, k, v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
        )
    
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    out.backward(dout)
    dqkv = qkv.grad

    (dq_ref, dk_ref, dv_ref,) = torch.autograd.grad(out_ref, (q, k, v), dout)
    
    print(f"dQ max diff: {(dqkv[:,:,0] - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dqkv[:,:,1] - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dqkv[:,:,2] - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dqkv[:,:,0] - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dqkv[:,:,1] - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dqkv[:,:,2] - dv_ref).abs().mean().item()}")

