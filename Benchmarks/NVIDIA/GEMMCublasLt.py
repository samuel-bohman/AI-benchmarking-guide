import json
import os
import shlex
import subprocess
import datetime
import time
import csv
from Infra import tools
from prettytable import PrettyTable

class GEMMCublastLt:
    def __init__(self, path: str, machine: str, b: int = 1, i: int = 1000, w: int = 10000):
        self.name = "GEMMCublasLt"
        config = self.get_config(path)
        self.m, self.n, self.k, self.duration, self.datatype = self.config_conversion(config)
        self.b = b
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
        self.buffer = []

        # A100 does not support fp8
        if "A100" in machine:
            self.datatype = "fp16"

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

    def build(self):
        bindir = tools.create_dir("bin")
        self.bindir = bindir
        path = "superbenchmark"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/gitaumark/superbenchmark",
                    path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tools.write_log(tools.check_error(results))
        current = os.getcwd()
        build_path = os.path.join(
            current,
            "superbenchmark/superbench/benchmarks/micro_benchmarks/cublaslt_gemm",
        )
        os.chdir(build_path)

        results = subprocess.run(
            ["cmake", "-S", "./"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        tools.write_log(tools.check_error(results))

        results = subprocess.run(
            ["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        tools.write_log(tools.check_error(results))
        print(results.stderr.decode('utf-8'))
        results = subprocess.run(
            ["mv", "cublaslt_gemm", bindir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.chdir(current)

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        print("Running CublasLt with datatype " + self.datatype + "...")
        current = os.getcwd()
        if self.datatype == "fp8e4m3":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 192, 768]
        elif self.datatype == "fp4e2m1":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 802816, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 192, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 192, 768]
        else:
            m_dims = [1024, 2048, 4096, 8192, 16384, 1024, 6144, 802816, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 2145, 12288, 192, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 1024, 12288, 192, 768]
        os.chdir(self.bindir)
        buffer = []
        for i in range(len(m_dims)):
            results = subprocess.run(
                [
                    "./cublaslt_gemm",
                    "-m",
                    str(m_dims[i]),
                    "-n",
                    str(n_dims[i]),
                    "-k",
                    str(k_dims[i]),
                    "-b",
                    str(self.b),
                    "-i",
                    str(self.i),
                    "-w",
                    str(self.w),
                    "-t",
                    self.datatype,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log = results.stdout.decode('utf-8').split()
            buffer.append(log)
            tools.write_log(tools.check_error(results))
        table1 = PrettyTable()

        with open('../Outputs/GEMMCublasLt_Performance_' + self.machine_name + '_' + self.datatype+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["M", "N", "K", "Batch", "Time(us)", "TFLOPS"])
            table1.field_names = ["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"]
            for item in buffer:
                writer.writerow(item)
                table1.add_row(item)
        print(table1)
        os.chdir(current)
