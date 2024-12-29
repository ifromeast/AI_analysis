# DeepSpeed-Ulysses 原理与实现


DeepSpeed-Ulysses attention 的原理比较简单，即：
- 输入沿序列维度切分到多个 GPU 上，每个 GPU 只处理序列的一部分
- 在计算 Attention 前，通过 all2all 操作，将序列维度的划分转化到注意力头上，这样每个 GPU 都有完整的序列长度，但只有部分注意力头（类似于TP）
- Attention 计算完成后，再通过 all2all 操作，将注意力头维度的划分转化回序列维度，这样每个 GPU 都有完整的注意力头，但只有部分序列

其原理如下图所示：

![alt text](./images/ulysses.png)

当然以上过程也可以简单表示为：

![alt text](./images/ulysses_simple.png)


接下来，我们详细解释一下 Ulysses 的实现细节。其中最关键的就是 all2all 操作的实现

对于第一个 all2all 操作，即 (bs, seqlen/P, hc, hs) --> (bs, seqlen, hc/P, hs) 实现如下：
```python
# input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
bs, shard_seqlen, hc, hs = input.shape
seqlen = shard_seqlen * seq_world_size
shard_hc = hc // seq_world_size

# transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
# (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
input_t = (input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous())

output = torch.empty_like(input_t)
# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
# (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

if seq_world_size > 1:
    dist.all_to_all_single(output, input_t, group=group)
    if use_sync:
        torch.cuda.synchronize()
else:
    output = input_t
# if scattering the seq-dim, transpose the heads back to the original dimension
output = output.reshape(seqlen, bs, shard_hc, hs)

# (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
```

对于第二个 all2all 操作，即 (bs, seqlen, hc/P, hs) --> (bs, seqlen/P, hc, hs) 实现如下：
```python
# input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
bs, seqlen, shard_hc, hs = input.shape
hc = shard_hc * seq_world_size
shard_seqlen = seqlen // seq_world_size
seq_world_size = dist.get_world_size(group)

# transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
# (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
input_t = (
    input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
    .transpose(0, 3)
    .transpose(0, 1)
    .contiguous()
    .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
)

output = torch.empty_like(input_t)
# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
# (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
if seq_world_size > 1:
    dist.all_to_all_single(output, input_t, group=group)
    if use_sync:
        torch.cuda.synchronize()
else:
    output = input_t

# if scattering the seq-dim, transpose the heads back to the original dimension
output = output.reshape(hc, shard_seqlen, bs, hs)

# (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
```

可通过 `./ulysses/test_ulysses_attn.py` 验证精度。

接下来，我们记录一下这种方式下的性能数据（8*4090）
| batch_size | seq_len | nheads | head_size | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 2 | 4096 | 16 | 128 | True | 487.977 | 2.049 | 188.1 | 67.1 |
| 2 | 8192 | 16 | 128 | True | 239.528 | 4.175 | 376.2 | 131.7 |
| 2 | 16384 | 16 | 128 | True | 103.540 | 9.658 | 752.5 | 227.6 |
| 2 | 32768 | 16 | 128 | True | 25.509 | 39.201 | 1505 |  313.0 |
| 2 | 65536 | 16 | 128 | True | 13.264 | 75.394 | 3010 | 466.6 |
| 2 | 128000 | 16 | 128 | True | 4.720 | 211.875 | 5891.9 | 633.5 |
| 2 | 4096 | 16 | 128 | False | 231.360 | 4.322 | 188.1 | 111.3 |
| 2 | 8192 | 16 | 128 | False | 111.034 | 9.006 | 376 | 213.6 |
| 2 | 16384 | 16 | 128 | False | 45.489 | 21.983 | 752.5 | 350.1 |
| 2 | 32768 | 16 | 128 | False | 10.379 | 96.347 | 1505 | 319.5 |
| 2 | 65536 | 16 | 128 | False | 4.640 | 215.524 |  3010 | 571.4 |
| 2 | 128000 | 16 | 128 | False | 1.685 | 593.629 | 5891.9 | 791.3 |


DeepSpeed Ulysses 的特点：
- all2all 通信方式效率高
- 并行度不能超过注意力头的数量


# 参考资料
[1] https://github.com/feifeibear/long-context-attention


