### This is a condensed plotting script to plot one file's output ###

import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_values(file_path):
    global_steps, train_losses, train_times = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"global_step: (\d+) .* reduced_train_loss: ([\d.]+) .* train_step_timing in s: ([\d.]+)", line)
            if match:
                global_steps.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_times.append(float(match.group(3)))
    return os.path.basename(file_path), (global_steps, train_losses, train_times)

def analyze_steady_state_comprehensive(data, window_size=20, std_threshold=0.01, min_consistent_windows=3):
    for file_name, (steps, _, times) in data.items():
        times = np.array(times)
        steps = np.array(steps)
        consistent = 0
        start_idx = None

        for i in range(len(times) - window_size + 1):
            std = np.std(times[i:i + window_size])
            if std < std_threshold:
                consistent += 1
                if consistent >= min_consistent_windows:
                    start_idx = i - (min_consistent_windows - 1)
                    break
            else:
                consistent = 0

        if start_idx is not None:
            steady_times = times[start_idx:]
            steady_steps = steps[start_idx:]
            avg, std = np.mean(steady_times), np.std(steady_times)
            print(f"  • {file_name}: Avg = {avg:.4f}s ± {std:.4f} over steps {steady_steps[0]}–{steady_steps[-1]}")
        else:
            print(f"  • {file_name}: Could not detect stable steady state.")

def plot_specific_file(data, metric="time", lower_x=None, upper_x=None):
    for file_name, (steps, losses, times) in data.items():
        steps = np.array(steps)
        values = np.array(times if metric == "time" else losses)
        label = "Training Time" if metric == "time" else "Loss"

        if lower_x is not None and upper_x is not None:
            mask = (steps >= lower_x) & (steps <= upper_x)
            steps = steps[mask]
            values = values[mask]

        plt.figure(figsize=(10,5))
        plt.plot(steps, values, marker='o', label=file_name)
        plt.xlabel("Global Step")
        plt.ylabel(label)
        plt.title(f"{label} vs Global Step")
        plt.grid()
        plt.legend()

        os.makedirs("Plots", exist_ok=True)
        out_file = f"Plots/{metric}_{file_name.replace('.txt','')}.png"
        plt.savefig(out_file, dpi=300)
        print(f"Plot saved to {out_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    key, data_tuple = extract_values(args.file)
    data = {key: data_tuple}

    plot_specific_file(data, metric="time") # plot the time metric for the file
    analyze_steady_state_comprehensive(data) # analyze the steady state behavior for the time and print this value out

if __name__ == "__main__":
    main()
