import os
import sys
import subprocess
import torch

from Benchmarks.NVIDIA import GEMMCublasLt as gemm
from Benchmarks.NVIDIA import HBMBandwidth as HBM
from Benchmarks.NVIDIA import NVBandwidth as NV
from Benchmarks.NVIDIA import NCCLBandwidth as NCCL
from Benchmarks.NVIDIA import FlashAttention as FA
from Benchmarks.NVIDIA import FIO
from Benchmarks.NVIDIA import CPUStream as CPU
from Benchmarks.NVIDIA import Multichase as Multichase
from Benchmarks.NVIDIA import LLMBenchmark as llmb
from Infra import tools


machine_name = ""
current = os.getcwd()
tools.create_dir("Outputs")

def get_system_specs():
    file = open("Outputs/system_specs.txt", "w")

    results = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,vbios_version,driver_version,memory.total", "--format=csv"], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output = results.stdout.decode('utf-8').split('\n')[1].split(",")
    file.write("GPU name     : "+ output[0]+"\n")
    file.write("VBIOS    : "+ output[1]+"\n")
    file.write("driver version   : "+ output[2]+"\n")
    file.write("GPU memory capacity  : "+ output[3]+"\n")
    
    results = subprocess.run("nvcc --version | grep release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    cuda_version = results.stdout.decode('utf-8').split(",")[1].strip().split(" ")[1]
    file.write("CUDA version     : "+cuda_version+"\n")

    if output[0].strip() != "NVIDIA Graphics Device" or "GB200" in output[0]:
        results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        ubuntu = results.stdout.decode('utf-8').strip().split("\t")[1]
        file.write("ubuntu version   : "+ubuntu+"\n")
        file.write("pytorch version  : {torch.__version__}\n")

    file.close()
    return output[0].strip()

def run_CublasLt():
    test = gemm.GEMMCublastLt("config.json",machine_name) 
    test.build()
    test.run_model_sizes()
   
def run_HBMBandwidth():
    if machine_name == "NVIDIA Graphics Device" or "GB200" in machine_name:
        print("HBM bandwidth Test not supported on this machine")
        return
    test = HBM.HBMBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_NVBandwidth():
    test = NV.NVBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_NCCLBandwidth():
    test = NCCL.NCCLBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention("config.json", machine_name)
    test.run()

def run_Multichase():
    test = Multichase.Multichase("config.json", machine_name)
    test.build()
    test.run()

def run_CPUStream():
    test = CPU.CPUStream("config.json", machine_name)
    test.build()
    test.run()
    
def run_FIO():
    test = FIO.FIO("config.json", machine_name)
    test.run()
    
def run_LLMBenchmark():
    if machine_name == "NVIDIA Graphics Device" or "GB200" in machine_name:
        print("LLMBenchmark Test not supported on this machine")
        return
    test = llmb.LLMBenchmark("config.json", current, machine_name)
    test.install_requirements()
    test.prepare_datasets()
    test.download_models()
    test.run_benchmark()

machine_name = get_system_specs()
arguments = []
match = False
for arg in sys.argv:
    arguments.append(arg.lower())

if ("gemm" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)
    
if ("nccl" in arguments):
    match = True
    run_NCCLBandwidth()
    os.chdir(current)
    
if ("hbm" in arguments):
    match = True
    run_HBMBandwidth()
    os.chdir(current)
    
if ("nv" in arguments):
    match = True
    run_NVBandwidth()
    os.chdir(current)
    
if ("fa"  in arguments):
    match = True
    run_FlashAttention()
    os.chdir(current)

if ("multichase" in arguments):
    match = True
    run_Multichase()
    os.chdir(current)

if ("cpustream" in arguments):
    match = True
    run_CPUStream()
    os.chdir(current)
    
if ("fio" in arguments):
    match = True
    run_FIO()
    os.chdir(current)
    
if ("llm" in arguments):
    match = True
    run_LLMBenchmark()
    os.chdir(current)
    
if ("all" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)
    run_NCCLBandwidth()
    os.chdir(current)
    run_Multichase()
    os.chdir(current)
    run_CPUStream()
    os.chdir(current)
    run_HBMBandwidth()
    os.chdir(current)
    run_NVBandwidth()
    os.chdir(current)
    run_FlashAttention()
    os.chdir(current)
    run_FIO()
    run_LLMBenchmark()
if not match: 
    print("Usage: python3 NVIDIA_runner.py [arg]\n   or: python3 NVIDIA_runner.py [arg1] [arg2] ... to run more than one test e.g python3 NVIDIA_runner.py hbm nccl\nArguments are as follows, and are case insensitive:\nAll tests:  all\nCuBLASLt GEMM:  gemm\nNCCL Bandwidth: nccl\nHBMBandwidth:   hbm\nNV Bandwidth:   nv\nFlash Attention: fa\nFIO Tests:   fio\nLLM Inference Workloads: llm\nCPU Stream: cpustream\nMultichase:  multichase")
    
