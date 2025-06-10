import os
import datetime
import subprocess
pwd = os.getcwd() + "/Outputs/log.txt"

def create_dir(name: str):
    current = os.getcwd()
    outdir = os.path.join(str(current), name)
    isdir = os.path.isdir(outdir)
    if not isdir:
        os.mkdir(outdir)
    return outdir

def write_log(message: str, filename: str = pwd):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]\n {message}\n"

    with open(filename, "a") as file:
        file.write(log_entry)

def check_error(results):
    if results.stderr:
        return results.stderr.decode("utf-8")
    return results.stdout.decode("utf-8")

def get_hostname():
    results = subprocess.run(["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if results.stderr:
        return ""
    return results.stdout.decode("utf-8").strip()
