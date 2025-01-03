# USP 性能基准

好了，接下来可以改变参数记录下USP性能基准(设备为 8 * 4090D)(bs=2, nheads=16, head_size=16)，如下表所示：

- seq_len 与 ring attention type 的影响 

| ring type | seq_len | ulysses_degree | ring_degree | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| basic | 4096 | 2 | 4 | True | 84.019 | 11.902 | 240 | 11.5 |
| basic | 8192 | 2 | 4 | True | 84.852 | 11.785 | 480 | 46.6 |
| basic | 16384 | 2 | 4 | True | 44.255 | 22.596 | 960 | 97.3 |
| basic | 32768 | 2 | 4 | True | 12.837 | 77.898 | 1920 | 112.9 |
| basic | 65536 | 2 | 4 | True | 6.444 | 155.191 | 3841 | 226.7 |
| basic | 128000 | 2 | 4 | True | 2.644 | 378.174 | 7518.9 | 354.9 |
| stripe | 4096 | 2 | 4 | True | 59.643 | 16.767 | 244.1 | 8.2 |
| stripe | 8192 | 2 | 4 | True | 48.455 | 20.638 | 488.3 | 26.6 |
| stripe | 16384 | 2 | 4 | True | 31.839 | 31.408 | 976.7 | 70.0 |
| stripe | 32768 | 2 | 4 | True | 17.832 | 56.078 | 1953.5 | 156.8 |
| stripe | 65536 | 2 | 4 | True | 6.540 | 152.911 | 3907.0 | 230.1 |
| stripe | 128000 | 2 | 4 | True | 3.184 | 314.057 | 7638.8 | 427.3 |
| zigzag | 4096 | 2 | 4 | True | 71.727 | 13.942 | 240.1 | 9.8 |
| zigzag | 8192 | 2 | 4 | True | 44.264 | 22.592 | 480.3 | 24.3 |
| zigzag | 16384 | 2 | 4 | True | 32.001 | 31.249 | 960.5 | 70.4 |
| zigzag | 32768 | 2 | 4 | True | 17.036 | 58.700 | 1921 | 149.8 |
| zigzag | 65536 | 2 | 4 | True | 6.026 | 165.935 | 3842.0 | 212.0 |
| zigzag | 128000 | 2 | 4 | True | 3.123 | 320.243 | 7514.0 | 419.1 |


- seq_len 与 degree 的影响 

| ring type | seq_len | ulysses_degree | ring_degree | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| zigzag | 4096 | 2 | 4 | True | 71.727 | 13.942 | 240.1 | 9.8 |
| zigzag | 8192 | 2 | 4 | True | 44.264 | 22.592 | 480.3 | 24.3 |
| zigzag | 16384 | 2 | 4 | True | 32.001 | 31.249 | 960.5 | 70.4 |
| zigzag | 32768 | 2 | 4 | True | 17.036 | 58.700 | 1921 | 149.8 |
| zigzag | 65536 | 2 | 4 | True | 6.026 | 165.935 | 3842.0 | 212.0 |
| zigzag | 128000 | 2 | 4 | True | 3.123 | 320.243 | 7514.0 | 419.1 |
| zigzag | 4096 | 4 | 2 | True | 203.134 | 4.923 | 240.1 | 27.9 |
| zigzag | 8192 | 4 | 2 | True | 102.374 | 9.768 | 480.3 | 56.3 |
| zigzag | 16384 | 4 | 2 | True | 51.234 | 19.518 | 960.5 | 112.7 |
| zigzag | 32768 | 4 | 2 | True | 19.720 | 50.709 | 1921 | 173.5 |
| zigzag | 65536 | 4 | 2 | True | 8.613 | 116.098 | 3842.0 | 303.1 |
| zigzag | 128000 | 4 | 2 | True | 3.639 | 274.807 | 7514.0 | 488.4 |


- seq_len 与 反向 的影响 

| ring type | seq_len | ulysses_degree | ring_degree | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| zigzag | 4096 | 2 | 4 | True | 71.727 | 13.942 | 240.1 | 9.8 |
| zigzag | 8192 | 2 | 4 | True | 44.264 | 22.592 | 480.3 | 24.3 |
| zigzag | 16384 | 2 | 4 | True | 32.001 | 31.249 | 960.5 | 70.4 |
| zigzag | 32768 | 2 | 4 | True | 17.036 | 58.700 | 1921 | 149.8 |
| zigzag | 65536 | 2 | 4 | True | 6.026 | 165.935 | 3842.0 | 212.0 |
| zigzag | 128000 | 2 | 4 | True | 3.123 | 320.243 | 7514.0 | 419.1 |
| zigzag | 4096 | 2 | 4 | False | 31.615 | 31.631 | 240.1 | 15.2 |
| zigzag | 8192 | 2 | 4 | False | 20.200 | 49.505 | 480.3 | 38.8 |
| zigzag | 16384 | 2 | 4 | False | 10.513 | 95.121 | 960.5 | 80.9 |
| zigzag | 32768 | 2 | 4 | False | 5.714 | 175.013 | 1921 | 175.9 |
| zigzag | 65536 | 2 | 4 | False | 2.134 | 468.519 | 3842.0 | 262.8 |
| zigzag | 128000 | 2 | 4 | False | 1.170 | 854.512 | 7514.0 | 549.7 |


# loongtrain double ring 性能基准

接下来记录一下 Loongtrain double ring attention 的性能基准(设备为 8 * 4090D)(bs=2, nheads=16, head_size=16)，如下表所示：

| ring type | seq_len | context_size | window_size | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| zigzag | 4096 | 8 | 4 | True | 57.112 | 17.509 | 228.1 | 7.8 |
| zigzag | 8192 | 8 | 4 | True | 46.554 | 21.480 | 456.3 | 25.6 |
| zigzag | 16384 | 8 | 4 | True | 30.797 | 32.471 | 912.7 | 67.7 |
| zigzag | 32768 | 8 | 4 | True | 17.982 | 55.610 | 1825.5 | 158.2 |
| zigzag | 65536 | 8 | 4 | True | 7.416 | 134.849 | 3651.0 | 260.9 |
| zigzag | 128000 | 8 | 4 | True | 3.270 | 305.810 | 7145.8 | 438.8 |
| zigzag | 4096 | 8 | 4 | False | 29.410 | 34.002 | 228.1 | 14.1 |
| zigzag | 8192 | 8 | 4 | False | 18.230 | 54.853 | 456.4 | 35.1 |
| zigzag | 16384 | 8 | 4 | False | 10.783 | 92.736 | 912.7 | 83.0 |
| zigzag | 32768 | 8 | 4 | False | 5.580 | 179.200 | 1825.5 | 171.8 |
| zigzag | 65536 | 8 | 4 | False | 1.774 | 563.702 | 3651.0 | 218.4 |
| zigzag | 128000 | 8 | 4 | False | 0.998 | 1001.561 | 7145.8 | 469.0 |


- 与 window_size 的关系

| ring type | seq_len | context_size | window_size | fwd_only | throughput(iters/s) | latency(ms/iter) | peak memory(MB/device) | speed(TFLOPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| zigzag | 4096 | 8 | 4 | True | 71.676 | 13.952 | 228.1 | 9.8 |
| zigzag | 8192 | 8 | 4 | True | 52.277 | 19.129 | 456.3 | 28.7 |
| zigzag | 16384 | 8 | 4 | True | 31.828 | 31.419 | 912.7 | 69.9 |
| zigzag | 32768 | 8 | 4 | True | 17.982 | 50.553 | 1825.5 | 173.9 |
| zigzag | 65536 | 8 | 4 | True | 8.186 | 122.157 | 3651.0 | 288.0 |
| zigzag | 128000 | 8 | 4 | True | 3.206 | 311.917 | 7144.8 | 430.3 |
