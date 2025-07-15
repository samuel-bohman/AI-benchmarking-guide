import subprocess
import os
import docker
from Infra import tools

class FlashAttention:
    def __init__(self, path:str, machine: str):

        self.name='FlashAttention'
        self.machine_name = machine
        self.dir_path = path
        self.container = None

        # self.buffer = []

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'name': 'flash_attention',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined', 'apparmor=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {
                str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'},
                '/mnt/resource_nvme/hf_cache': {'bind': '/root/.cache/huggingface', 'mode': 'rw'}
            },
            'environment': {
                'HUGGINGFACE_HUB_CACHE': '/mnt/resource_nvme/hf_cache'
            },
            'tty': True,
            'detach': True,
            'auto_remove': True,
            'shm_size': '16G'
        }

        # Creates new Docker container
        print("Pulling docker container rocm/vllm:latest...")
        self.container = client.containers.run('rocm/vllm:latest', **docker_run_options)
        print(f"Created Docker Container ID: {self.container.id}")

    def run(self):
        self.create_container()
        print("Installing Flash Attention in the container...")
        install_result = self.container.exec_run('pip install flash-attn --no-build-isolation')
        if install_result.exit_code != 0:
            print(f"Failed to install Flash Attention: {install_result.output.decode('utf-8')}")
            self.container.kill()
            return

        print("Running Flash Attention...")
        res = self.container.exec_run(f'python3 {self.dir_path}/flash-attention/benchmarks/benchmark_flash_attention.py | grep -A 2 "batch_size=2, seqlen=8192 ###"')
        tools.write_log(res.output.decode('utf-8'))
        print(res.output.decode('utf-8'))

        self.container.kill()

        with open(self.dir_path + "/Outputs/FlashAttention_" + self.machine_name + ".txt", "w") as file:
            file.write(res.output.decode('utf-8'))
