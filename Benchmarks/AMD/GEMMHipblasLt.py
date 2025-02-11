import json
import os
from Infra import tools
from prettytable import PrettyTable
import docker

class GEMMHipBLAS:
    def __init__(self, path: str, dir_path: str, machine: str, i: int = 1000, w: int = 10000):
        self.name = "GEMMHipBLAS"
        config = self.get_config(path)
        self.m, self.n, self.k, self.duration, self.datatype = self.config_conversion(config)
        self.dir_path = dir_path
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
        self.container = None

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def parse_json(self, config, var):
        if var == "duration":
            return config["inputs"]["duration"]
        if var == "datatype":
            return config["inputs"]["datatype"]
        start = config["inputs"][var]["start"]
        end = config["inputs"][var]["end"]
        interval = config["inputs"][var]["interval"]
        data = [a for a in range(start, end, interval)]
        if not data or data[-1] < end:
            data.append(end)
        return data

    def config_conversion(self, config):
        m = self.parse_json(config, "m")
        n = self.parse_json(config, "n")
        k = self.parse_json(config, "k")
        duration = self.parse_json(config, "duration")
        datatype = self.parse_json(config, "datatype")
        return m, n, k, duration, datatype

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

        # Creates new Docker container 
        self.container = client.containers.run('rocm/vllm-dev:main', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")

    def build(self):
        path = "hipBLASLt"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/hipBLASLt " + self.dir_path + "/hipBLASLt"
            results = self.container.exec_run(clone_cmd, stderr=True)
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/hipBLASLt && git checkout a11ccf64efcd818106dbe37768f69dfcc0a7ff22"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return

            results = self.container.exec_run(f'sudo apt-get -y update', stderr=True)
            tools.write_log(results.output.decode('utf-8'))
            results = self.container.exec_run(f'sudo apt -y install llvm-dev', stderr=True)
            tools.write_log(results.output.decode('utf-8'))
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/hipBLASLt && ./install.sh -dc -a gfx942"', stderr=True)
            tools.write_log(results.output.decode('utf-8'))

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        print("Running HipBLAS...")
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]

        for i in range(len(m_dims)):
            hipblas_cmd = 'cd ' + self.dir_path + '/Benchmarks/AMD && ./hipBLAS_runner.sh ' + str(m_dims[i]) + ' ' +  str(n_dims[i]) + ' ' + str(k_dims[i])
            results = self.container.exec_run(f'/bin/sh -c ' + '"' + hipblas_cmd + '"')
            tools.write_log(results.output.decode('utf-8'))

        with open(self.dir_path + '/Outputs/GEMMHipBLAS_results.txt', 'r') as resFile:
            table1 = PrettyTable()
            table1.field_names = ["M","N","K","TFLOPS"]
            for line in resFile:
                l = line.strip()
                if l[0] == "T":
                    l = l.split(',')
                    m = l[4]
                    n = l[5]
                    k = l[6]
                    tflops = float(l[-3])/1000
                    table1.add_row([m,n,k,tflops])

        print(table1)
        self.container.kill()
