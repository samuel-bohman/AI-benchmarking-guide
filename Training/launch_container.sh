### This bash script launches the docker container interactively and runs the pretraining file and logs its output to a log file. ###


#!/bin/bash

# would need to change the log directory to your own directory
LOG_DIR="/shared/home/haffaticati/mdhekial/llama-model/nemo-logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/pretraining_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

sudo docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /shared/home/haffaticati/mdhekial/llama-model:/workspace/nemo-run \ # would also need to change the mounting for the docker container
  nvcr.io/nvidia/nemo:25.02 \ # assume have the docker container available
  bash -c "cd /workspace/nemo-run && python run_llama3_8b_pretrain.py" \
  | tee "$LOGFILE"
