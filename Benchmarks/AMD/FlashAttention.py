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

        self.buffer = []
    
    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'name': 'flash_attention',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True,
            'auto_remove': True
        }

        # Creates new Docker container
        self.container = client.containers.run('powderluv/vllm_dev_channel:20240927', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")
 
    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run('git clone https://github.com/Dao-AILab/flash-attention.git',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
        
        build_path = os.path.join(current, 'flash-attention')
        os.chdir(build_path)

        results = subprocess.run('git checkout 418d677',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        #results = subprocess.run('GPU_ARCHS="gfx942" python3 setup.py install',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.create_container()
        print("Running Flash Attention...")
        #res = self.container.exec_run(f'/bin/sh -c cd {self.dir_path}/flash-attention')
        res = self.container.exec_run(f'python3 {self.dir_path}/flash-attention/benchmarks/benchmark_flash_attention.py | grep -A 2 "batch_size=2, seqlen=8192 ###"')
        tools.write_log(res.output.decode('utf-8'))
        
        self.container.kill()

        file = open(self.dir_path + "/Outputs/FlashAttention_" + self.machine_name + ".txt", "w")
        file.write(res.output.decode('utf-8'))
