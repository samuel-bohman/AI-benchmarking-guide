import docker
import os
import json
import csv
from prettytable import PrettyTable
import json

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.dir_path = dir_path
        self.precision = "float16"
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
        self.container = client.containers.run('powderluv/vllm_dev_channel:20240927', **docker_run_options)

        print(f"Docker Container ID: {self.container.id}")


    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    for batch_size in self.config['models'][model_name]['batch_sizes']:
                        for input_size in  self.config['models'][model_name]['input_length']:
                            for output_size in self.config['models'][model_name]['output_length']:
                                warmup = self.config['models'][model_name]['warmup']
                                number_of_runs = self.config['models'][model_name]['number_of_runs']
                                print(f"Benchmarking {model_name} | TP Size: {tp_size} | Batch Size: {batch_size} | Input Size: {input_size} | Output Size: {output_size}")



                                print(model_name, tp_size, batch_size, input_size, output_size)
                                run_benchmark_command = f'''
                                    /bin/bash -c \
                                    "python /app/vllm/benchmarks/benchmark_latency.py \
                                        --model amd/{model_name} \
                                        --dtype {self.precision} \
                                        --kv-cache-dtype fp8 \
                                        --gpu-memory-utilization 0.99 \
                                        --quantization fp8 \
                                        --distributed-executor-backend mp \
                                        --max-model-len 8192 \
                                        --tensor-parallel-size {tp_size} \
                                        --num-iters {number_of_runs} \
                                        --num-iters-warmup {warmup} \
                                        --num-scheduler-steps 10 \
                                        --input-len {input_size} \
                                        --output-len {output_size} \
                                        --batch-size {batch_size}"
                                    '''

                                rb1 = self.container.exec_run(run_benchmark_command)
                                if rb1.exit_code != 0:
                                    print(rb1.output.decode('utf-8'))
                                    self.container.kill()
                                    return

                                temp = rb1.output.decode('utf-8').split('\n')
                                for line in temp:
                                    if "Avg latency" in line:
                                        res = float(line.split(' ')[2])
                                        result = round(int(batch_size) * int(output_size) / res, 2)
                                        table1 = PrettyTable()
                                        table1.add_row(['Model Name', model_name])
                                        table1.add_row(['Input/Output lengths', str(input_size) + "/" + str(output_size)])
                                        table1.add_row(['World Size (TP size)', str(tp_size)])
                                        table1.add_row(['Batch Size', str(batch_size)])
                                        table1.add_row(['Throughput (tokens/sec)', str(result)])
                                        table1.add_row(['Latency (ms)', str(round(res * 1000, 2))])

                                        print(table1.get_string(header=False))
                                        self.save_data([model_name, str(input_size), str(output_size), str(tp_size), str(batch_size), str(result), str(res * 1000)], 'Outputs/LLMBenchmark_' + self.machine + '.csv')

        self.container.kill()


    def save_data(self, data, file_path):

        # Check if the file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode if it exists, otherwise create it
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Model_name", "Input_length", "Output_length", "TP_size", "Batch_size", "Tokens per sec", "Latency(ms)"])
            writer.writerow(data)