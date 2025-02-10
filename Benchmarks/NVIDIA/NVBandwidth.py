import json
import subprocess
import os
from Infra import tools

class NVBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NVBandwidth'
        self.machine_name = machine
        config = self.get_config(path)
        self.num_runs, self.interval = self.config_conversion(config)
        self.buffer = []
    
    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def parse_json(self, config):
        return config['inputs']['num_runs'], config['inputs']['interval']
    
    def config_conversion(self, config)->tuple[list, list, list]:
        return self.parse_json(config)
        
    def build(self):
        current = os.getcwd()
        path ='nvbandwidth'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nvbandwidth', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        build_path = os.path.join(current, 'nvbandwidth')
        os.chdir(build_path)
        results = subprocess.run(['sed', '-i', '2i\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)', 'CMakeLists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results = subprocess.run(['sudo', './debian_install.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))       
        os.chdir(current)

    def run(self): 
        current = os.getcwd()
        os.chdir(os.path.join(current, 'nvbandwidth'))
        print("Running NVBandwidth...")
        buffer=[] 
        results = subprocess.run('./nvbandwidth -t device_to_host_memcpy_ce host_to_device_memcpy_ce device_to_device_bidirectional_memcpy_read_ce', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        log = results.stdout.decode('utf-8')
        buffer.append(log)
        os.chdir(current)
    
        file = open("Outputs/NVBandwidth_" + self.machine_name + ".txt", "w")
        for item in buffer:
            file.write(item)
            print(item)
     
        self.buffer=buffer
        os.chdir(current)
