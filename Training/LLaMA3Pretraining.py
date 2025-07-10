import os
import re
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Infra import tools
from datetime import datetime

class LLaMA3Pretraining:
    def __init__(self, config_path: str, machine_name: str):
        self.name = "LLaMA3Pretraining"
        self.machine_name = machine_name
        self.config = self.get_config(config_path) # get config path from JSON
        self.mount_path = self.config.get("mount_path", ".") # mount docker container
        self.training_script = self.config.get("training_script", "Training/ExecuteLLaMA3Pretrain.py")
        self.container = self.config.get("docker_image", "nvcr.io/nvidia/nemo:25.04")

    def get_config(self, path: str):
        with open(path) as f:
            data = json.load(f)
        try:
            return data[self.name]
        except KeyError:
            raise KeyError(f"{self.name} section not found in config")
        
    def plot_results(self, file_path: str = None):
        # extract values from the output file
        global_steps, train_losses, train_times = [], [], [] 
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"global_step: (\d+) .* reduced_train_loss: ([\d.]+) .* train_step_timing in s: ([\d.]+)", line)
                if match:
                    global_steps.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    train_times.append(float(match.group(3)))
        
        # calculate steady state time value
        window_size = 20
        std_threshold = 0.01
        min_consistent_windows = 3

        consistent = 0
        start_idx = None
        for i in range(len(train_times) - window_size + 1):
            std = np.std(train_times[i:i + window_size])
            if std < std_threshold:
                consistent += 1
                if consistent >= min_consistent_windows:
                    start_idx = i - (min_consistent_windows - 1)
                    break
            else:
                consistent = 0

        if start_idx is not None:
            steady_times = train_times[start_idx:]
            steady_steps = global_steps[start_idx:]
            steady_state = np.mean(steady_times)
            tools.write_log(f"Steady-state detected: {steady_state:.4f}s over steps {steady_steps[0]}–{steady_steps[-1]}")
        else:
            tools.write_log("Could not detect a stable steady state.")
            steady_state = 0.0

        # create grid for both loss and time plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.tight_layout(pad=4.0)

        # plot loss
        ax1.plot(global_steps, train_losses, marker='o', label="Training Loss")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss vs Global Step")
        ax1.grid(True)
        ax1.legend()

        # plot time
        ax2.plot(global_steps, train_times, marker='o', color='orange', label="Training Time (s)")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Training Time vs Global Step")
        ax2.grid(True)
        ax2.legend()

        # add text annotation for the steady state value
        fig.text(
            0.5,         # x-position 
            0.03,        # y-position 
            f"Steady state value: {steady_state}",
            ha='center',
            fontsize=12,
            style='italic',
        )

        # save to outputs folder
        plot_path = f"Outputs/LLaMA3_8B_Pretrain_Results"
        plt.savefig(plot_path, dpi=300)
        print(f"Training loss and time plot with steady state saved to {plot_path}")
        tools.write_log(f"Training loss and time plot with steady state saved to {plot_path}") # print to log.txt
        plt.close()


    def run(self):
        log_path = f"Outputs/log.txt" # log to log file
        tools.write_log(f"Launching NeMo container for {self.machine_name}.") # write to log file
        print(f"Launching NeMo container for {self.machine_name} and logging at 'Outputs/log.txt'.") # also let the user know where log is

        command = [
            "sudo", "docker", "run", "--rm", "-i",
            "--gpus", "all",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-v", f"{self.mount_path}:/workspace/nemo-run",
            self.container,
            "bash", "-c", f"cd /workspace/nemo-run && python {self.training_script}"
        ]

        # launch command and write to log file (this shows all info about epoch, training time, etc.)
        with open("Outputs/log.txt", "w") as file:
            subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, text=True)
        
        # now plot the results 
        print(f"Pretraining has finished with output saved to: {log_path}. Now plotting.")
        self.plot_results(log_path)
