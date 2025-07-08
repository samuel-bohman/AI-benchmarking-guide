import os
import json
import subprocess
from Infra import tools
from datetime import datetime

class LLaMA3Pretraining:
    def __init__(self, config_path: str, machine_name: str):
        self.name = "LLaMA3Pretraining"
        self.machine_name = machine_name
        self.config = self.get_config(config_path)
        self.log_dir = self.config.get("log_dir", "./logs")
        self.mount_path = self.config.get("mount_path", ".")
        self.training_script = self.config.get("training_script", "ExecuteLLaMA3Pretrain.py")
        self.container = self.config.get("docker_image", "nvcr.io/nvidia/nemo:25.04")
        self.outputs_dir = "Outputs"

    def get_config(self, path: str):
        with open(path) as f:
            data = json.load(f)
        try:
            return data[self.name]
        except KeyError:
            raise KeyError(f"{self.name} section not found in config")

    def build(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        tools.write_log(f"Output and log directories ensured.")

    def run(self):
        print(f"Launching NeMo container for {self.machine_name}...")

        output_path = os.path.join(self.outputs_dir, f"LLaMA3Pretraining_{self.machine_name}.txt")
        gpu_logfile = os.path.join(self.outputs_dir, f"LLaMA3Pretraining_GPU_{self.machine_name}.csv")

        # Compose the command to start GPU logging and training
        bash_command = (
            f'echo "[INFO] Starting GPU logging to {gpu_logfile}" && '
            f'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.mem,power.draw,memory.used '
            f'--format=csv,nounits,noheader -l 10 > "{gpu_logfile}" & '
            'GPULOG_PID=$! && '
            f'python {self.training_script}; '
            'kill $GPULOG_PID'
        )

        command = [
            "sudo", "docker", "run", "--rm", "-i",
            "--gpus", "all",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-v", f"{self.mount_path}:/workspace/nemo-run",
            self.container,
            "bash", "-c", f"cd /workspace/nemo-run && {bash_command}"
        ]

        results = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = results.stdout.decode("utf-8")

        with open(output_path, "w") as file:
            file.write(output)

        tools.write_log(tools.check_error(results))
        print(f"Output saved to: {output_path}")
        print(f"GPU log saved to: {gpu_logfile}")

        # now use the plotting code to plot the output files (this will plot the file and also print out steady state metrics)
        plot_script = os.path.join(os.path.dirname(__file__), "LLaMA3Plotter.py")
        output_path = os.path.join(self.outputs_dir, f"LLaMA3Pretraining_{self.machine_name}.txt")
        subprocess.run(["python3", plot_script, "--file", output_path])

