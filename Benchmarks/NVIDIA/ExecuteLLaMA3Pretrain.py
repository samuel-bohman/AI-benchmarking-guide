import json
import subprocess
import time
import numpy as np
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import (
    fp16_mixed,
    fp16_with_fp8_mixed
)
from nemo.lightning.pytorch.callbacks.callback import Callback

def load_config():
    with open("config.json") as f:
        return json.load(f)["LLaMA3Pretraining"]


def configure_recipe(cfg, nodes=1, gpus_per_node=4):
    precision = cfg.get("precision", "fp16").lower()
    plugin = fp16_with_fp8_mixed() if precision == "fp8" else fp16_mixed()

    recipe = llm.llama3_8b.pretrain_recipe(
        dir="/checkpoints/llama3",
        name="llama3_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    # Parallelism
    tp = cfg["parallelism"]["tp"]
    pp = cfg["parallelism"]["pp"]
    vp = cfg["parallelism"].get("vp")
    cp = cfg["parallelism"]["cp"]

    recipe.model.config.tensor_model_parallel_size = tp
    recipe.model.config.pipeline_model_parallel_size = pp
    recipe.model.config.virtual_pipeline_model_parallel_size = vp

    recipe.trainer.strategy.tensor_model_parallel_size = tp
    recipe.trainer.strategy.pipeline_model_parallel_size = pp
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = vp
    recipe.trainer.strategy.context_parallel_size = cp

    recipe.data.micro_batch_size = cfg["micro_batch_size"]

    recipe.trainer.plugins = plugin
    recipe.trainer.accelerator = "gpu"
    recipe.trainer.devices = gpus_per_node

    # just for testing purposes to see if the plot generates properly
    recipe.trainer.max_steps = 3

    return recipe


def local_executor_torchrun(nodes=1, devices=4):
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    return run.LocalExecutor(
        ntasks_per_node=devices,
        launcher="torchrun",
        env_vars=env_vars
    )


def run_pretraining():
    cfg = load_config()
    recipe = configure_recipe(cfg)

    executor = local_executor_torchrun(
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices
    )

    run.run(recipe, executor=executor, name="llama3_8b_pretraining")


if __name__ == "__main__":
    run_pretraining()
