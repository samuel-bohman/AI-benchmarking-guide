import os
import json
import subprocess
from Infra import tools
from datetime import datetime

class LLaMA3Pretraining:
    def __init__(self, config_path: str, machine_name: str):
        self.name = "LLaMA3Pretraining"
        self.machine_name = machine_name
        self.config = self._load_config(config_path)
        self.log_dir = self.config.get("log_dir", "./logs")
        self.mount_path = self.config.get("mount_path", ".")
        self.training_script = self.config.get("training_script", "ExecuteLLaMA3Pretrain.py")
        self.container = self.config.get("docker_image", "nvcr.io/nvidia/nemo:25.04")
        self.outputs_dir = "Outputs"

    def _load_config(self, path: str):
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

        command = [
            "docker", "run", "--rm", "-i",
            "--gpus", "all",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-v", f"{self.mount_path}:/workspace/nemo-run",
            self.container,
            "bash", "-c", f"cd /workspace/nemo-run && python {self.training_script}"
        ]

        results = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = results.stdout.decode("utf-8")

        with open(output_path, "w") as file:
            file.write(output)

        tools.write_log(tools.check_error(results))
        print(f"Output saved to: {output_path}")

        # now use the plotting code to plot the output files (this will plot the file and also print out steady state metrics)
        plot_script = os.path.join(os.path.dirname(__file__), "LLMA3Plotter.py")
        output_path = os.path.join(self.outputs_dir, f"LLaMA3Pretraining_{self.machine_name}.txt")
        subprocess.run(["python", plot_script, "--file", output_path])

