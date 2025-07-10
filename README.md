# Azure AI Benchmarking Guide

Inefficient workload optimization can significantly increase operational costs for customers, making it essential to define clear performance benchmarks. This benchmarking guide establishes performance standards across a series of microbenchmarks, tests, and language models. These results are designed to help Azure users maximize efficiency, identify bottlenecks, and fine-tune resource allocation on Azure. By providing detailed performance insights, this guide ensures users can optimize workloads for both cost and performance, improving overall cloud infrastructure utilization. The guide currently supports the following SKUs:

### NVIDIA SKUs
- ND A100 v4
- ND H100 v5
- ND H200 v5.
- ND GB200 v6

### AMD SKUs
- ND MI300X v5

## Tests Included - NVIDIA

### 1. Microbenchmark - CublasLt GEMM
The [CuBLASLt General Matrix-to-matrix Multiply](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/GEMMCublasLt.py) (GEMM) is a performance evaluation test for the CUDA Basic Linear Algebra Subroutines (CuBLAS) library for matrix and vector operations that leverages the parallel processing capabilities of GPUs. The benchmark is designed to assess the speed of matrix-to-matrix multiplication, which is the fundamental operation in AI applications, by measuring for varying matrix sizes (m, n, and k). The results shown below are with random initialization (best representation of real-life workloads) and datatype FP8.

In the guide, we run CuBLASLt on various matrix sizes. See the [`run_model_sizes`](https://github.com/Azure/AI-benchmarking-guide/blob/b2d64036d7ba6d4171e0f1ece26967dc97c7740f/Benchmarks/NVIDIA/GEMMCublasLt.py#L244) function in `GEMMCublasLt.py`.

### 2. Microbenchmark - NCCL Bandwidth

The [NCCL bandwidth test](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/NCCLBandwidth.py) is a benchmark provided by NVIDIA's NCCL (NVIDIA Collective Communications Library) library. NCCL is a high-performance library, designed to accelerate interGPU communication, that optimizes communication between multiple GPUs within a single node or across multiple nodes in a multi-GPU system.
The performance measured is the data transfer bandwidth between GPUs using various communication patterns, such as point-to-point (pairwise) communication or collective communication (communication between multiple GPUs).

### 3. Microbenchmark - HBM Bandwidth
[High Bandwidth Memory](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/HBMBandwidth.py) (HBM) is designed to provide a significant boost in memory bandwidth for GPUs by handling vast amounts of data through vertical stacking of multiple layers of memory chips, connected by through-silicon vias.

### 4. Microbenchmark - NV Bandwidth
The [NV Bandwidth](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/NVBandwidth.py) benchmark measures the bandwidth achieved while transferring packets CPU-to-GPU and GPU-to-CPU over PCIe, and GPU-to-GPU over NVLink.

### 5. Microbenchmark - Flash Attention
[FlashAttention](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/FlashAttention.py) is an algorithm to speed up attention and reduce the memory footprint for Natural Language Models—without any approximation. It is meant to speed up training and inference by reordering the attention computation and leveraging classical techniques (tiling, recomputation) to reduce memory usage from quadratic to linear in sequence length.

### 6. End-to-end Inference Workloads
To assess how different system components (as tested by the microbenchmarks) affect overall performance, we suggetsing running some [end-to-end workloads](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/LLMBenchmark.py). The models we used for benchmarking are the current industry standards across various sizes: LLAMA 3 (8B, 70B, and 405B). The performance of the model inferencing (throughput) is measured in tokens per second, accounting for both processing input tokens and generating output tokens. The workloads run in a TensorRT-LLM environment. Users need huggingface credentials to download all the model weigths. Visit [huggingface.co](https://huggingface.co/) to create an account and obtain access to the models. After obtaining your credentials, use `huggingface-cli login` to input your access token.

### 7. CPU STREAM Benchmark 
The [CPU STREAM](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/CPUStream.py) benchmark measures memory bandwidth performance rather than raw CPU speed. It evaluates how efficiently a system can move data between the CPU and RAM, which is crucial for memory-intensive applications like scientific computing and HPC.

### 8. Multichase Benchmark 
The [Multichase](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/Multichase.py) benchmark is a memory latency benchmark designed to measure pointer-chasing latency in a system. Unlike traditional memory benchmarks like STREAM, which focus on memory bandwidth, Multichase is used to evaluate random memory access latency, which is crucial for workloads that rely on irregular memory access patterns, such as databases and graph processing.

## Tests Included - AMD

### 1. Microbenchmark - hipBLAS GEMM
The [hipBLASLt General Matrix-to-matrix Multiply](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/GEMMHipblasLt.py) (GEMM) is a performance evaluation test for the hip Basic Linear Algebra Subroutines (hipBLAS) library for matrix and vector operations that leverages the parallel processing capabilities of GPUs. The benchmark is designed to assess the speed of matrix-to-matrix multiplication, which is the fundamental operation in AI applications, by measuring for varying matrix sizes (m, n, and k). The results shown below are with random initialization (best representation of real-life workloads) and datatype FP8.


### 2. Microbenchmark - RCCL Bandwidth
The [RCCL bandwidth test](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/RCCLBandwidth.py) is a benchmark provided by ROCm RCCL (ROCm Collective Communications Library) library. RCCL is a high-performance library, designed to accelerate interGPU communication, that optimizes communication between multiple GPUs within a single node or across multiple nodes in a multi-GPU system.
The performance measured is the data transfer bandwidth between GPUs using various communication patterns, such as point-to-point (pairwise) communication or collective communication (communication between multiple GPUs).

### 3. Microbenchmark - HBM Bandwidth
[High Bandwidth Memory](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/HBMBandwidth.py) (HBM) is designed to provide a significant boost in memory bandwidth for GPUs by handling vast amounts of data through vertical stacking of multiple layers of memory chips, connected by through-silicon vias.

### 4. Microbenchmark - TransferBench
The [NV Bandwidth](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/TransferBench.py) benchmark measures the bandwidth achieved while transferring packets CPU-to-GPU and GPU-to-CPU.

### 5. Microbenchmark - Flash Attention
[FlashAttention](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/FlashAttention.py) is an algorithm to speed up attention and reduce the memory footprint for Natural Language Models—without any approximation. It is meant to speed up training and inference by reordering the attention computation and leveraging classical techniques (tiling, recomputation) to reduce memory usage from quadratic to linear in sequence length.

### 6. End-to-end Inference Workloads
To assess how different system components (as tested by the microbenchmarks) affect overall performance, we suggetsing running some [end-to-end workloads](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/LLMBenchmark.py). The models we used for benchmarking are the current industry standards across various sizes: Mistral (7B parameters), LLAMA 3 (8B, 70B, and 405B). The performance of the model inferencing (throughput) is measured in tokens per second, accounting for both processing input tokens and generating output tokens. The workloads run in a vLLM environment. Users need huggingface credentials to download all the model weigths. Visit [huggingface.co](https://huggingface.co/) to create an account and obtain access to the models. After obtaining your credentials, use `huggingface-cli login` to input your access token.


# HOW TO RUN THE BENCHMARKS

Installation of benchmark dependencies requires multiple steps.
A convenience script `install-dependencies.sh` is provided to simplify installation. On GB200, make sure to run this script inside a virtual environment.

```bash
install-dependencies.sh 
```

### NVIDIA

**ND GB200 v6** still runs an externally managed environment, so the benchmarks need to be ran inside a python virtual env. Start a venv with the following commands before running the benchmarks:

```
python3 -m venv venv && source venv/bin/activate
```

Usage: `python3 NVIDIA_runner.py [arg]`\
   or: `python3 NVIDIA_runner.py [arg1] [arg2]` ... to run more than one test e.g `python3 NVIDIA_runner.py hbm nccl`\
Arguments are as follows, and are case insensitive:\
All tests:   `all`\
CuBLASLt GEMM:   `gemm`\
NCCL Bandwidth:  `nccl`\
HBMBandwidth:    `hbm`\
NV Bandwidth:   `nv`\
Flash Attention: `fa`\
FIO Tests:   `fio`\
CPU Stream: `cpustream`\
Multichase:  `multichase`\
LLM Inference Workloads: `llm`


### AMD
Usage: `python3 AMD_runner.py [arg]`\
   or: `python3 AMD_runner.py [arg1] [arg2]` ... to run more than one test e.g `python3 AMD_runner.py hbm nccl`\
Arguments are as follows, and are case insensitive:\
All tests:  `all`\
HipBLAS GEMM:  `gemm`\
RCCL Bandwidth: `rccl`\
HBMBandwidth:   `hbm`\
TransferBench:   `transfer`\
Flash Attention: `fa`\
FIO Tests:   `fio`\
LLM Inference Workloads: `llm`

### Extras
- The console output and errors are logged in `Outputs/log.txt.`
- The file [`config.json`](https://github.com/Azure/AI-benchmarking-guide/blob/main/config.json) contains the specific settings for the benchmarks.
- The [`models`](https://github.com/Azure/AI-benchmarking-guide/blob/b2d64036d7ba6d4171e0f1ece26967dc97c7740f/config.json#L83) field in `config.json` contains all the inference models that can be benchmarked. To run benchmark for a specific model, set `use_model: true`. They are all set to `false` by default.
- All the AMD models in `config.json` are marked with `"type": "amd"`
- All the NVIDIA models in `config.json` are marked with `"type": "nvidia"`
- Test results will be stored in the `Outputs` directory.

You can find example of results for the ND A100 v4, ND H100 v5 and ND H200 v5 virtual machines stored under [`Azure_Results`](https://github.com/Azure/AI-benchmarking-guide/tree/main/Azure_Results).

### Storage
- The benchmarks, especially LLM Benchmarks, take up a lot of space. Therefore, we recommend cloning this benchmark repository onto a mounted NVMe disk. 
- Some of the AMD benchmarks are ran in docker containers, which are automatically created and killed when the tests are ran. To make sure that these docker containers don't fill up your storage space, change the default location that docker stores its files. Do this by:

  `vim /etc/docker/daemon.json`

  add a `data-root` field to `daemon.json` and make sure it points to a directory on the mounted NVMe disk, or somewhere with sufficient storage space (at least 1TB). for example:

  ```
  {
       "data-root":"/mnt/resource_nvme/docker", 
  }
  ```
  note that this won't be the path to the NVMe disk on all machines, so check to make sure.
