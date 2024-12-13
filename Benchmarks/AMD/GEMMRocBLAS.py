import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from prettytable import PrettyTable
import docker


class GEMMRocBLAS:
    def __init__(self, path: str, dir_path: str, machine: str, i: int = 1000, w: int = 10000):
        self.name = "GEMMRocBLAS"
        config = self.get_config(path)
        self.m, self.n, self.k, self.duration, self.datatype = self.config_conversion(config)
        self.dir_path = dir_path
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
        self.buffer = []
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

        #if existing container exists
        # self.container = client.containers.get('c34fc0616f7a')

        # Creates new Docker container from https://hub.docker.com/r/rocm/pytorch/tags
        self.container = client.containers.run('rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue', **docker_run_options)

        print(f"Docker Container ID: {self.container.id}")

    def build(self):

        results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/Benchmarks/AMD/rocBLAS_Bench && hipcc -o rocblas_gemm rocblas_gemm.cpp -I/opt/rocm/include -I/opt/rocm/include/rocblas -I/opt/rocm/include/hiprand -L/opt/rocm/lib -lrocblas -lhiprand -lhsa-runtime64 -lamdhip64 -fopenmp -D__HIP_PLATFORM_AMD__"', stderr=True)

        if results.exit_code != 0:
            print(results.output.decode('utf-8'))
            return
        else:
            print("Successfully built target rocblas_gemm")


       



    # if no shmoo data found, run shmoo. if found, use existing shmoo data
    def run_shmoo(self):
        if os.path.isfile('Outputs/GEMMRocBLAS_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv'):
            self.buffer = self.parse_csv('Outputs/GEMMRocBLAS_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv')
        else:
            self.buffer = self.run()

        self.plot_shmoo()


    # run GEMM Sweep, start and end dims can be altered in config.json
    # results saved in Outputs folder
    def run(self) -> list:
        print("Running GEMM Sweep...")
        current = os.getcwd()

        end_interval = str(self.m[-1])

        buffer = []
        with open(self.dir_path + '/Outputs/GEMMRocBLAS_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["M", "N", "K", "Batch", "Time(us)", "TFLOPS"])

            for i in range(len(self.m)):
                for j in range(len(self.n)):
                    for t in range(len(self.k)):
                        a = str(self.m[i]) == end_interval
                        b = str(self.n[j]) == end_interval
                        c = str(self.k[t]) == end_interval

                        if (a and b) or (b and c) or (a and c):
                            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/Benchmarks/AMD/rocBLAS_Bench && ./rocblas_gemm -m {str(self.m[i])} -n {str(self.n[j])} -k {str(self.k[t])} -i {str(self.i)} -w {str(self.w)}"')
                          
                            if results.exit_code != 0:
                                print(results.output.decode('utf-8'))
                                self.container.kill()
                                return
                              
                            log = results.output.decode("utf-8")
                            # print("m: ", self.m[i] ," n: ", self.n[j], " k: ", self.k[t])
                            # handle errors and failed cases
                            buffer.append(log.split())
                            writer.writerow(log.split())

        os.chdir(current)
        self.container.kill()
        return buffer


    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        print("Running RocBLAS...")
        current = os.getcwd()


        m_dims = [1024, 2048, 4096, 8192, 16384, 1024, 6144, 802816, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 2145, 12288, 192, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 1024, 12288, 192, 768]

        buffer = []

        for i in range(len(m_dims)):
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/Benchmarks/AMD/rocBLAS_Bench && ./rocblas_gemm -m {str(m_dims[i])} -n {str(n_dims[i])} -k {str(k_dims[i])} -i {str(self.i)} -w {str(self.w)}"')
            if results.exit_code != 0:
                print(results.output.decode('utf-8'))
                self.container.kill()
                return
            
            log = results.output.decode("utf-8").split()
            buffer.append(log)
        self.container.kill()

        table1 = PrettyTable()

        with open(self.dir_path + '/Outputs/GEMMRocBLAS_Performance_' + self.machine_name + '_' + self.datatype+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["M", "N", "K", "Batch", "Time(us)", "TFLOPS"])
            table1.field_names = ["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"]
            for item in buffer:
                writer.writerow(item)
                table1.add_row(item)

        print(table1)



    def plot_shmoo(self):
        arr = np.array(self.buffer)


        # splitting up the data into m, n, k sweeps
        m_arr = []
        n_arr = []
        k_arr = []


        # size of the other 2 dims that are constant
        dim_size = '4096'
        for i in range(len(arr)):
            if arr[i][1] == dim_size and arr[i][2] == dim_size:
                m_arr.append(arr[i])
            if arr[i][0] == dim_size and arr[i][2] == dim_size:
                n_arr.append(arr[i])
            if arr[i][0] == dim_size and arr[i][1] == dim_size:
                k_arr.append(arr[i])


        m_arr = np.array(m_arr)
        n_arr = np.array(n_arr)
        k_arr = np.array(k_arr)

        # plot M Shmoo
        x = m_arr[:, 0].astype(int)
        y = m_arr[:, 5].astype(float)


        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM M Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("M Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMRocBLAS M Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close(fig)

        # plot N Shmoo
        x = n_arr[:, 1].astype(int)
        y = n_arr[:, 5].astype(float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM N Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("N Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMRocBLAS N Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close(fig)

        # plot K shmoo
        x = k_arr[:, 2].astype(int)
        y = k_arr[:, 5].astype(float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM K Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("K Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMRocBLAS K Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close()
