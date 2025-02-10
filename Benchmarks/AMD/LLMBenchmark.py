import docker
import os
import json
import csv
from prettytable import PrettyTable
import json
from Infra import tools

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.dir_path = dir_path
        self.precision = "half"
        self.container = None
        self.machine = machine

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'entrypoint':'/bin/bash',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'environment': {'HF_HOME': str(self.dir_path)},
            'tty': True,
            'detach': True
        }

        # Creates new Docker container
        self.container = client.containers.run('rocm/vllm-dev:20241121-tuned', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "amd":
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    for max_num_seq in self.config['models'][model_name]['max_num_seqs']:
                        for input_size in self.config['models'][model_name]['input_length']:
                            for output_size in self.config['models'][model_name]['output_length']:
                                for request in self.config['models'][model_name]['num_requests']:
                                    print(f"Benchmarking {model_name} | TP Size: {tp_size} | Input Size: {input_size} | Output Size: {output_size}")
                                    run_benchmark_command = f'''
                                        /bin/bash -c \
                                        "python /app/vllm/benchmarks/benchmark_throughput.py \
                                            --model amd/{model_name} \
                                            --quantization fp8 \
                                            --kv-cache-dtype fp8 \
                                            --dtype half \
                                            --gpu-memory-utilization 0.90 \
                                            --distributed-executor-backend mp \
                                            --num-scheduler-steps 10 \
                                            --tensor-parallel-size {tp_size} \
                                            --enable-chunked-prefill false \
                                            --max-seq-len-to-capture 131072 \
                                            --max-num-batched-tokens 131072 \
                                            --max-model-len 8192 \
                                            --max-num-seqs {max_num_seq} \
                                            --num-prompts {request} \
                                            --input-len {input_size} \
                                            --output-len {output_size}"
                                        '''

                                    rb1 = self.container.exec_run(run_benchmark_command)
                                    
                                    tools.write_log(rb1.output.decode('utf-8'))
                                    self.container.kill()

                                    temp = rb1.output.decode('utf-8').split('\n')
                                    for line in temp:
                                        if "Throughput: " in line:
                                            result = line.split(' ')[6]
                                            table1 = PrettyTable()
                                            table1.add_row(['Model Name', model_name])
                                            table1.add_row(['Input/Output lengths', str(input_size) + "/" + str(output_size)])
                                            table1.add_row(['World Size (TP size)', str(tp_size)])
                                            table1.add_row(['Throughput (tokens/sec)', str(result)])

                                            print(table1.get_string(header=False))
                                            self.save_data([model_name, str(input_size), str(output_size), str(tp_size), str(result)], 'Outputs/LLMBenchmark_' + self.machine + '.csv')

        self.container.kill()

    def save_data(self, data, file_path):
        file_exists = os.path.exists(file_path)
        # Open the file in append mode if it exists, otherwise create it
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Model_name", "Input_length", "Output_length", "TP_size", "Tokens per sec"])
            writer.writerow(data)
