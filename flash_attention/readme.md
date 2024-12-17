# Flash Attention 及其修改版的精度及性能基准

## 1. 标准 Flash Attention
标准 Flash Attention 提供了 3 个最基本的 Attention kernel，分别是：
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












# 参考资料
[1] https://github.com/feifeibear/long-context-attention

[2] https://github.com/zhuzilin/ring-flash-attention

[3] https://github.com/Dao-AILab/flash-attention