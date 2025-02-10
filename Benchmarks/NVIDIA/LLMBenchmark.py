import os
from Infra import tools
import subprocess
import json
from prettytable import PrettyTable
import json
from huggingface_hub import snapshot_download

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.dir_path = dir_path
        self.machine = machine

        tools.create_dir(self.dir_path + "/datasets")
        tools.create_dir(self.dir_path + "/engines")
        tools.create_dir(self.dir_path + "/hub")

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")
    
    def install_requirements(self):
        # Install required packages
        print("Installing Required Packages")
        i2 = subprocess.run("apt-get -y install libopenmpi-dev", shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(i2))
        i2 = subprocess.run("pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs", shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(i2))
 
        os.environ['HF_HOME'] = self.dir_path
        os.environ['LD_LIBRARY_PATH'] = "/home/azureuser/.local/lib/python3.10/site-packages/tensorrt_libs/:/home/azureuser/.local/lib/python3.10/site-packages/tensorrt_llm/libs" + os.environ['LD_LIBRARY_PATH']

        # Clone TensorRT-LLM repo
        if not os.path.exists(os.path.join(self.dir_path, 'TensorRT-LLM')):
            print("Cloning TensorRT-LLM reopsitory from https://github.com/NVIDIA/TensorRT-LLM.git") 
            i4 = subprocess.run("git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM && git checkout v0.15.0", shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(i4))

    def download_models(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                snapshot_download(repo_id=model_name, cache_dir=self.dir_path+"/hub")

    def prepare_datasets(self):
        for model_name in self.config['models']:
            max_isl = 0
            max_osl = 0
            max_sum = 0
            max_dataset_path = ""
            if self.config['models'][model_name]['use_model']:
                for i in range(len(self.config['models'][model_name]['input_sizes'])): 
                    isl = self.config['models'][model_name]['input_sizes'][i]
                    osl = self.config['models'][model_name]['output_sizes'][i]
                    name = model_name.split('/')[1]
                    if (isl + osl > max_sum):
                        max_sum = isl + osl
                        max_isl = isl
                        max_osl = osl
                        max_dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(max_isl) + "_" + str(max_osl) + ".txt" 

                    dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(isl) + "_" + str(osl) + ".txt"
                    prepare_dataset_command = f'''
                        python3 {self.dir_path}/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
                        --stdout \
                        --tokenizer {model_name} \
                        token-norm-dist \
                        --num-requests {self.config['models'][model_name]['num_requests']} \
                        --input-mean {isl} \
                        --output-mean {osl} \
                        --input-stdev=0 \
                        --output-stdev=0 > {dataset_path}
                        '''                
                    
                    be2 = subprocess.run(prepare_dataset_command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    tools.write_log(tools.check_error(be2)) 
                
                if not os.path.exists(self.dir_path + "/engines/" + model_name):
                    print("Building engine for ", model_name)
                    build_engine_command = f'''
                        trtllm-bench \
                        --workspace {self.dir_path + "/engines"} \
                        --model {model_name} build \
                        --tp_size {self.config['models'][model_name]['tp_size']} \
                        --dataset {max_dataset_path} \
                        --quantization {self.config['models'][model_name]['precision']} 
                        ''' 
                    
                    be2 = subprocess.run(build_engine_command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    tools.write_log(tools.check_error(be2))

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                print("Benchmarking ", model_name)
                for i in range(len(self.config['models'][model_name]['input_sizes'])):               
                    isl = self.config['models'][model_name]['input_sizes'][i]
                    osl = self.config['models'][model_name]['output_sizes'][i]
                    tp = self.config['models'][model_name]['tp_size']
                    name = model_name.split('/')[1]
                   
                    dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(isl) + "_" + str(osl) + ".txt"
                    results_path = self.dir_path + "/Outputs/results_" + name + "_" + str(isl) + "_" + str(osl) + ".txt"
                    
                    run_benchmark_command = f'''
                        trtllm-bench \
                        --model {model_name} throughput\
                        --dataset {dataset_path} \
                        --engine_dir {self.dir_path + "/engines/" + model_name + "/tp_" + str(tp) + "_pp_1"} > {results_path}
                        '''                
                    
                    be2 = subprocess.run(run_benchmark_command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    tools.write_log(tools.check_error(be2))
