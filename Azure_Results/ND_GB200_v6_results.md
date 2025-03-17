# Azure ND GB200 v6 Benchmark Results

## System Specifications

| GPU           | NVIDIA GB200 185GB |
|---------------|-------------------|
| CPU           | ARM Neoverse-V2 |
| Ubuntu        |   24.04  |
| CUDA          |   12.8  |
| NVIDIA Driver | 570.99  |
| VBIOS         | 97.00.82.00.30 |
| NCCL          |    2.25.1 |
| PyTorch       |    2.6.0   |


## Microbenchmarks
### GEMM CuBLASLt 

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND GB200 v6 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 349.4                  |
| 2048        | 2048      | 2048     | 1675.8                |
| 4096        | 4096      | 4096     | 2578.9                |
| 8192        | 8192      | 8192     | 2624.5                 |
| 16384       | 16384     | 16384    | 2695.0                 |
| 32768       | 32768     | 32768    | 2623.1                 |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 729.1                  |
| 6144        | 12288     | 12288    | 2709.9                 |
| 802816      | 192       | 768      | 1430.9                  |

### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND GB200 v6 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 147.6   |
| Flash Attention 2.0   | 373.9  |

### NV Bandwidth

|                       | ND GB200 v6 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 201                |
| Device to Host        | 193                |
| Device to Device read |  1530              |


### NCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (Ring algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND GB200 v6 (GB/s) |
| -------------------- | ----------------- |
| 1K          | 0.10            |
| 2K          | 0.19            |
| 4K          | 0.38            |
| 8K          | 0.76            |
| 16K         | 1.46            |
| 32K         | 2.88            |
| 65K         | 5.71            |
| 132K        | 11.29           |
| 256K        | 22.41           |
| 524K        | 44.07           |
| 1M          | 80.48           |
| 2M          | 110.43          |
| 4M          | 146.27          |
| 8M          | 253.46          |
| 16M         | 331.09          |
| 33M         | 422.62          |
| 67M         | 550.73          |
| 134M        | 574.19          |
| 268M        | 593.53          |
| 536M        | 618.74          |
| 1G          | 646.87          |
| 2G          | 666.63          |
| 4G          | 673.06          |
| 8G          | 679.61          |

