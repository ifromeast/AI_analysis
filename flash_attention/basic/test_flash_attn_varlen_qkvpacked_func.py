import torch
from einops import rearrange, repeat
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
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

    assert nheads % nheads_k == 0
    # window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    q, k, v = qkv.clone().unbind(2)

    random_lengths = torch.randint(max(1, seqlen - 20), seqlen + 1, (batch_size, 1), device=device)
    key_padding_mask = (repeat(torch.arange(seqlen, device=device), "s -> b s", b=batch_size) < random_lengths)
    query_padding_mask = key_padding_mask.clone()

    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
    v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1).detach().requires_grad_()
    qkv = torch.stack([q, k, v], dim=2).detach().requires_grad_()

    output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen)
    dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen)

    cu_seqlens = cu_seqlens_q
    max_seqlen = max_seqlen_q

    out_unpad, sm_lse, S_dmask = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            return_attn_probs=True,
        )
    out = output_pad_fn(out_unpad)

    out_ref, attn_ref = attention_ref(
            q, k, v,
            query_padding_mask,
            key_padding_mask
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)

    (dqkv_unpad,) = torch.autograd.grad(out, qkv_unpad, g)
    dqkv = dqkv_pad_fn(dqkv_unpad)

    (dq_ref, dk_ref, dv_ref,) = torch.autograd.grad(out_ref, (q, k, v), g)

    print(f"dQ max diff: {(dqkv[:,:,0] - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dqkv[:,:,1] - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dqkv[:,:,2] - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dqkv[:,:,0] - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dqkv[:,:,1] - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dqkv[:,:,2] - dv_ref).abs().mean().item()}")

