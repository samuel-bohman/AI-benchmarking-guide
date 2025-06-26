import json
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import (
    fp16_mixed,
    fp16_with_fp8_mixed
)

def load_config():
    with open("config.json") as f:
        return json.load(f)["LLaMA3Pretraining"]

def configure_recipe(cfg, nodes=1, gpus_per_node=4):
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="/checkpoints/llama3",
        name="llama3_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    # Parallelism config
    recipe.model.config.tensor_model_parallel_size = cfg["parallelism"]["tp"]
    recipe.model.config.pipeline_model_parallel_size = cfg["parallelism"]["pp"]
    recipe.model.config.virtual_pipeline_model_parallel_size = cfg["parallelism"]["vp"]

    recipe.trainer.strategy.tensor_model_parallel_size = cfg["parallelism"]["tp"]
    recipe.trainer.strategy.pipeline_model_parallel_size = cfg["parallelism"]["pp"]
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = cfg["parallelism"]["vp"]
    recipe.trainer.strategy.context_parallel_size = cfg["parallelism"]["cp"]
    recipe.trainer.strategy.data_parallel_size = cfg["parallelism"]["dp"]

    recipe.data.micro_batch_size = cfg["micro_batch_size"]

    return recipe

def local_executor_torchrun(nodes=1, devices=4):
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    return run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

def run_pretraining():
    cfg = load_config()
    precision = cfg.get("precision", "fp16").lower()

    precision_plugin = (
        fp16_with_fp8_mixed() if precision == "fp8" else fp16_mixed()
    )

    recipe = configure_recipe(cfg)
    recipe.trainer = run.Config(
        nl.Trainer,
        plugins=precision_plugin,
        devices=recipe.trainer.devices,
        accelerator="gpu",
        strategy=recipe.trainer.strategy,
        callbacks=recipe.trainer.callbacks,
        logger=recipe.trainer.logger,
        precision="16",  # This stays "16" even for mixed FP8 usage
    )

    executor = local_executor_torchrun(
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices,
    )

    run.run(recipe, executor=executor, name="llama3_8b_pretraining")

if __name__ == "__main__":
    run_pretraining()