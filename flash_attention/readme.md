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

### 性能基准












# 参考资料
[1] https://github.com/feifeibear/long-context-attention

[2] https://github.com/zhuzilin/ring-flash-attention

[3] https://github.com/Dao-AILab/flash-attention