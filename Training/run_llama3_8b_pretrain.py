### This script runs the pre-configured llama3 8b model's pretraining recipe provided by the NVIDIA NeMo Framework ### 


import nemo_run as run

from nemo.collections import llm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 4):
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="/checkpoints/llama3", # Path to store checkpoints
        name="llama3_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.trainer.devices = 4 # specifying number of GPUs available
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 4) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    recipe = configure_recipe()
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor, name="llama3_8b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()
