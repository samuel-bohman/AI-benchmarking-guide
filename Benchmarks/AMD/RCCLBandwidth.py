import json
import docker
import os
import csv
import csv
from prettytable import PrettyTable
from Infra import tools

class RCCLBandwidth:
    def __init__(self, config_path:str, dir_path:str, machine: str):
        self.name='RCCLBandwidth'
        self.machine_name = machine
        config = self.get_config(config_path)
        self.start, self.end, self.num_gpus = self.config_conversion(config)
        self.dir_path = dir_path
        self.container = None
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
        return config['inputs']['start'], config['inputs']['end'], config['inputs']['num_gpus']

    def config_conversion(self, config)->tuple[list, list, list]:
        return self.parse_json(config)

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'entrypoint': '/bin/bash',
            'network': 'host',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True
        }

        # Creates new Docker container from https://hub.docker.com/r/rocm/pytorch/tags
        self.container = client.containers.run('rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")

    def build(self):
        path ='rccl'
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/rccl.git " + self.dir_path + "/rccl"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
 
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl && cmake . && make"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                
            results = self.container.exec_run(f'/bin/sh -c "cd .."', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

        path ='rccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/rccl-tests.git " + self.dir_path + "/rccl-tests"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl-tests && make HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl CUSTOM_RCCL_LIB={self.dir_path}/rccl/librccl.so && make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))            

    def run(self):
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]
        print("Running RCCL AllReduce...")
        for run in runs:
            run_cmd = "NCCL_ALGO=" + run + " " + self.dir_path +"/rccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 40 | grep float"
            run_cmd = '/bin/sh -c "' + run_cmd + '"'
            results = self.container.exec_run(run_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return            
            res = results.output.decode('utf-8').split('\n')
            log = []
            for line in res:
                line = line.split()
                if len(line) == 13:
                    log.append(line[11])
            buffer.append(log)

        table1 = PrettyTable()
        runs = ["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"]

        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])

        print(table1)
        self.buffer=buffer
        self.container.kill()
        self.save()


    def save(self):
        with open('Outputs/RCCLBandwidth_' + self.machine_name + '.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"])
            for i in range(len(self.buffer[0])):
                row = [self.buffer[0][i], self.buffer[1][i], self.buffer[2][i], self.buffer[3][i], self.buffer[4][i]]
                writer.writerow(row)
