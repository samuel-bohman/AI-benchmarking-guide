import json
import os
from Infra import tools
from prettytable import PrettyTable
import docker
import datetime

class GEMMHipBLASLt:
    def __init__(self, path: str, dir_path: str, machine: str, i: int = 1000, w: int = 10000):
        self.name = "GEMMHipBLASLt"
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
        # print("Pulling docker container rocm/vllm-dev:main...")
        # self.container = client.containers.run('rocm/vllm-dev:main', **docker_run_options)
        print("Pulling docker container rocm/vllm:latest...")
        self.container = client.containers.run('rocm/vllm:latest', **docker_run_options)
        print(f"Created Docker Container ID: {self.container.id}")

    def build(self):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Installing hipBLASLt package...")
        results = self.container.exec_run(f'/bin/sh -c "apt-get update && apt-get install -y hipblaslt"', stderr=True)
        if results.exit_code != 0:
            tools.write_log(results.output.decode('utf-8'))
            return

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run(self):
        print("Running HipBLASLt...")
        # self.m = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
        # self.n = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
        # self.k = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]

        results_file_path = self.dir_path + '/Outputs/GEMMHipBLASLt_results.txt'
        # Clear the results file before the run to avoid accumulating old results
        with open(results_file_path, 'w') as f:
            pass

        for i in range(len(self.m)):
            hipblas_cmd = f"cd {self.dir_path}/Benchmarks/AMD && ./hipBLASLt_runner.sh {self.m[i]} {self.n[i]} {self.k[i]}"
            results = self.container.exec_run(f'/bin/sh -c "{hipblas_cmd}"')
            tools.write_log(results.output.decode('utf-8'))

        with open(results_file_path, 'r') as resFile:
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
