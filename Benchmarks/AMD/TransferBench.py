import json
import os
import time
import subprocess
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

    def build(self):
        path = "TransferBench"
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building TransferBench...")
            clone_cmd = "git clone https://github.com/ROCm/TransferBench.git " + self.dir_path + "/TransferBench"
            results = subprocess.run(clone_cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            results = subprocess.run("mkdir " + self.dir_path + "/TransferBench/build && cd" + self.dir_path + "/TransferBench/build", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))

            results = subprocess.run("cd " + self.dir_path + "/TransferBench/build && CXX=/opt/rocm/bin/hipcc cmake .. && make", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            
    def run(self):
        print("Running TransferBench...")
        runs_executed = 0
        while runs_executed < self.num_runs:
            run_cmd = "sudo " + self.dir_path + "/TransferBench/build/TransferBench " + self.dir_path + "/Benchmarks/AMD/transferbench.cfg | grep -v '='"
            results = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            log = results.stdout.decode("utf-8")
            print(log)
            self.save(log, 'Outputs/TransferBench_' + self.machine_name + '.txt')
            runs_executed += 1
            time.sleep(int(self.interval))

    def save(self, data, filename):
        with open(filename, mode='w', encoding='utf-8') as file:
            file.write(data)
