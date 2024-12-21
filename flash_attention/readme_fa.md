# Flash Attention 的精度及性能基准

## 1. 标准 Flash Attention
标准 Flash Attention(本文采用的版本为flash-attn==2.6.3) 提供了 3 个最基本的 Attention kernel，分别是：
- `flash_attn_func`：最基本的Attention，直接使用 Q, K, V 作为输入
- `flash_attn_qkvpacked_func`：将 Q, K, V 合并成一个矩阵，然后直接使用这个矩阵作为输入
- `flash_attn_varlen_qkvpacked_func`：与 `flash_attn_qkvpacked_func` 类似，但是输入的 Q, K, V 的长度是可变的

### 精度验证
首先实现标准版的 Attention `AI_analysis/flash_attention/reference.py` 并验证以上3个函数与其精度的一致性，验证过程在单 GPU 上进行。运行代码如下：
```
cd AI_analysis/flash_attention
export PYTHONPATH=$(pwd)

python basic/test_flash_attn_func.py
python basic/test_flash_attn_qkvpacked_func.py
python basic/test_flash_attn_varlen_qkvpacked_func.py
```
输出结果如下：
```
Output max diff: 0.000244140625
Output mean diff: 7.510185241699219e-06
dQ max diff: 0.00048828125
dK max diff: 0.000244140625
dV max diff: 0.000244140625
dQ mean diff: 8.404254913330078e-06
dK mean diff: 8.285045623779297e-06
dV mean diff: 8.344650268554688e-06
```

### 分布式运行
验证单卡上的精度是重要的，但是也是不够的，因为实际使用中，我们通常需要多卡分布式运行。因此，我们还需要验证多卡上 Attention 的精度，验证过程在 8 卡上运行。运行代码如下：
```
torchrun --nproc_per_node=8 basic/test_flash_attn_qkvpacked_func_dist.py
```
输出结果如下：
```
##############################
# forward:
##############################
rank 4 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 6 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 2 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 3 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 5 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 1 out (distributed) - out_ref (non-distributed) diff: 0.001953125
rank 7 out (distributed) - out_ref (non-distributed) diff: 0.0009765625
rank 0 out (distributed) - out_ref (non-distributed) diff: 0.0078125
##############################
# backward:
##############################
dq diff:
Rank[0] max 0.00781, mean 0.000165
Rank[1] max 0.00195, mean 8.11e-05
Rank[2] max 0.00195, mean 6.29e-05
Rank[3] max 0.000977, mean 5.32e-05
Rank[4] max 0.000977, mean 4.74e-05
Rank[5] max 0.000977, mean 4.27e-05
Rank[6] max 0.000977, mean 3.98e-05
Rank[7] max 0.000488, mean 3.7e-05
dk diff:
Rank[0] max 0.0156, mean 0.000174
Rank[1] max 0.00195, mean 7.3e-05
Rank[2] max 0.00391, mean 5.2e-05
Rank[3] max 0.000977, mean 3.96e-05
Rank[4] max 0.000977, mean 3.05e-05
Rank[5] max 0.000488, mean 2.32e-05
Rank[6] max 0.000488, mean 1.67e-05
Rank[7] max 0.000488, mean 8.23e-06
dv diff:
Rank[0] max 0.0156, mean 0.000177
Rank[1] max 0.00195, mean 7.34e-05
Rank[2] max 0.000977, mean 5.2e-05
Rank[3] max 0.000977, mean 4.01e-05
Rank[4] max 0.000488, mean 3.1e-05
Rank[5] max 0.000488, mean 2.35e-05
Rank[6] max 0.000488, mean 1.68e-05
Rank[7] max 0.000244, mean 8.4e-06
```

### 性能基准
在确认了结果的正确性之后，我们还需要对性能进行基准测试。运行方式为：

```bash
torchrun --nproc_per_node=8 basic/benchmark_qkvpacked_func.py --batch_size 2 --seq_len 4096 --nheads 16 --head_size 128 --profile --fwd_only
```
首先分析一下 profile, 纯前向的 profile 如下所示：
![flash_attn_qkvpacked_func_bs_2_seq_4096_heads_16_d_128_rank_0_fwd_only_True](./pictures/flash_attn_qkvpacked_func_bs_2_seq_4096_heads_16_d_128_rank_0_fwd_only_True.png)

其中框出的即为一个 `flash_fwd_kernel` 的执行过程，在此参数下，执行周期为 1.165 ms。

同理可以得到包含反向阶段的 profile， 如下所示：
![flash_attn_qkvpacked_func_bs_2_seq_4096_heads_16_d_128_rank_0_fwd_only_False](./pictures/flash_attn_qkvpacked_func_bs_2_seq_4096_heads_16_d_128_rank_0_fwd_only_False.png)

其中在前向kernel `flash_fwd_kernel` 执行完成之后，会执行一个反向kernel `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`，执行周期为 2.797 ms。

好了，接下来可以改变参数记录下性能基准(设备为 8 张 4090D)：，如下表所示：
| batch_size | seq_len | nheads | head_size | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 2 | 4096 | 16 | 128 | True | 869.400 | 1.151 | 321 | 119.5 |
| 8 | 4096 | 16 | 128 | True | 245.441 | 4.052 | 1284 | 134.9 |
| 16 | 4096 | 16 | 128 | True | 126.438 | 7.931 | 2568 | 139.0 |
| 64 | 4096 | 16 | 128 | True | 32.027 | 31.196 | 10272 | 140.9 |
| 2 | 8192 | 16 | 128 | True | 241.179 | 4.090 | 642 | 132.6 |
| 2 | 16384 | 16 | 128 | True | 63.672 | 15.733 | 1284 | 140.0 |
| 2 | 32768 | 16 | 128 | True | 16.268 | 61.431 | 2568 | 143.1 |
| 2 | 4096 | 16 | 128 | False | 208.624 | 4.793 | 321 | 100.4 |
| 8 | 4096 | 16 | 128 | False | 65.390 | 15.293 | 1284 | 125.8 |
| 16 | 4096 | 16 | 128 | False | 33.022 | 30.283 | 2568 | 127.1 |
| 64 | 4096 | 16 | 128 | False | 8.260 | 121.060 | 10272 | 127.2 |
| 2 | 8192 | 16 | 128 | False | 67.940 | 14.719 | 642 | 130.7 |
| 2 | 16384 | 16 | 128 | False | 17.756 | 56.318 | 1284 | 136.7 |
| 2 | 32768 | 16 | 128 | False | 4.496 | 222.408 | 2568 | 138.4 |

由于 4090 上没有 NCCL 连接，故同时在 8*H100 上进行同样测试
| batch_size | seq_len | nheads | head_size | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 2 | 4096 | 16 | 128 | True | 1703.071 | 0.587 | 321 | 234.1 |
| 8 | 4096 | 16 | 128 | True | 517.433 | 1.933 | 1284 | 284.5 |
| 16 | 4096 | 16 | 128 | True | 277.761 | 3.600 | 2568 | 305.4 |
| 64 | 4096 | 16 | 128 | True | 73.597 | 13.588 | 10272 | 323.6 |
| 128 | 4096 | 16 | 128 | True | 37.115 | 26.943 | 20544 | 326.4 |
| 2 | 8192 | 16 | 128 | True | 502.247 | 1.991 | 642 | 276.1 |
| 2 | 16384 | 16 | 128 | True | 144.016 | 6.944 | 1284 | 316.6 |
| 2 | 32768 | 16 | 128 | True | 38.307 | 26.105 | 2568 | 336.9 |
| 2 | 65536 | 16 | 128 | True | 9.822 | 101.815 | 5136 | 345.5 |
| 2 | 4096 | 16 | 128 | False | 458.597 | 2.181 | 321 | 220.6 |
| 8 | 4096 | 16 | 128 | False | 135.901 | 7.358 | 1284 | 261.4 |
| 16 | 4096 | 16 | 128 | False | 69.625 | 14.363 | 2568 | 267.9 |
| 64 | 4096 | 16 | 128 | False | 17.851 | 56.019 | 10272 | 274.7 |
| 128 | 4096 | 16 | 128 | False | 8.927 | 112.020 | 20544 | 274.8 |
| 2 | 8192 | 16 | 128 | False | 137.236 | 7.287 | 642 | 264.1 |
| 2 | 16384 | 16 | 128 | False | 38.208 | 26.173 | 1284 | 294.1 |
| 2 | 32768 | 16 | 128 | False | 9.856 | 101.459 | 2568 | 303.4 |
| 2 | 65536 | 16 | 128 | False | 2.495 | 400.766 | 5136 | 307.2 |



# 参考资料
[1] https://github.com/feifeibear/long-context-attention

[2] https://github.com/zhuzilin/ring-flash-attention

[3] https://github.com/Dao-AILab/flash-attention