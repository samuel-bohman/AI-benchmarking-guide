import json
import os
import time
import docker
from Infra import tools

class TransferBench:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "TransferBench"
        self.machine_name = machine
        config = self.get_config(config_path)
        self.num_runs, self.interval = self.config_conversion(config)
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
        return config["inputs"]["num_runs"], config["inputs"]["interval"]

    def config_conversion(self, config) -> tuple[list, list, list]:
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
        path = "TransferBench"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/TransferBench.git " + self.dir_path + "/TransferBench"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return

            results = self.container.exec_run(f'/bin/sh -c "mkdir {self.dir_path}/TransferBench/build && cd {self.dir_path}/TransferBench/build"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/TransferBench/build && CXX=/opt/rocm/bin/hipcc cmake .. && make"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return
            else:
                print("Successfully built target TransferBench")

    def run(self):
        print("Running TransferBench...")
        runs_executed = 0
        while runs_executed < self.num_runs:
            run_cmd = self.dir_path + "/TransferBench/build/TransferBench " + self.dir_path + "/Benchmarks/AMD/transferbench.cfg"
            results = self.container.exec_run(run_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return
            log = results.output.decode("utf-8")
            print(log)
            self.save(log, 'Outputs/TransferBench_' + self.machine_name + '.txt')
            runs_executed += 1
            time.sleep(int(self.interval))

        self.container.kill()

    def save(self, data, filename):
        with open(filename, mode='w', encoding='utf-8') as file:
            file.write(data)
