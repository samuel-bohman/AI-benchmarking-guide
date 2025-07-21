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

        table1 = PrettyTable()
        table1.field_names = ["M", "N", "K", "TFLOPS"]
        total_tests = len(self.m)

        for i in range(total_tests):
            m_val, n_val, k_val = self.m[i], self.n[i], self.k[i]

            print(f"Running test {i+1}/{total_tests}: M={m_val}, N={n_val}, K={k_val}...")

            hipblas_cmd = f"cd {self.dir_path}/Benchmarks/AMD && ./hipBLASLt_runner.sh {m_val} {n_val} {k_val}"
            results = self.container.exec_run(f'/bin/sh -c "{hipblas_cmd}"')

            output = results.output.decode('utf-8').strip()
            tools.write_log(output)

            if results.exit_code == 0 and output:
                try:
                    # Parse the output line directly
                    line_parts = output.split(',')
                    tflops = float(line_parts[-2]) / 1000
                    table1.add_row([m_val, n_val, k_val, f"{tflops:.2f}"])
                    print(f"Result: {tflops:.2f} TFLOPS")
                except (IndexError, ValueError) as e:
                    table1.add_row([m_val, n_val, k_val, "Parse Error"])
                    print(f"Could not parse result for M={m_val}, N={n_val}, K={k_val}. Error: {e}")
            else:
                print(f"Test failed for M={m_val}, N={n_val}, K={k_val}. See logs for details.")

        print("\n--- Benchmark Summary ---")
        print(table1)
        self.container.kill()
