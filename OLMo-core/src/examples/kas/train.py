"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""
import os
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from pathlib import Path

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
    KASDataCollator
)

from olmo_core.data.numpy_dataset import (
    VSLCurriculumType,
    VSLCurriculumConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import (
    get_default_device, 
    seed_all
)

from olmo_core.train.common import Duration

# from metrics import MetricsLogger

# @dataclass
# class BatchMetricCallback(Callback):
#     """
#     Adds batch metrics.
#     """

#     def pre_train(self):
#         self.logger = MetricsLogger(self.trainer.save_folder)
#         self.batch_metrics = self.logger.load_batch_metrics()
#         self.dataset_metrics = self.logger.load_dataset_metrics()

#     def post_step(self):
#         if get_rank() != 0:
#             return
#         metrics = self.batch_metrics[self.step]
#         for name, value in metrics.items():
#             self.trainer.record_metric(f"train/{name}", value)
#             if name in self.dataset_metrics:
#                 value /= self.dataset_metrics[name]
#                 value = round(value, 4)
#                 self.trainer.record_metric(f"train/{name}_relative", value)

@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536

def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_config(config: Dict[str, Any]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    # ---- model ----
    model_name = config["model"]

    # ---- optim ----
    peak_lr = config["optim"]["lr"]
    weight_decay = config["optim"]["weight_decay"]

    # ---- dataset ----
    ds = config["dataset"]
    dataset_name = ds["name"]                      
    max_sequence_length = ds["max_sequence_length"]
    min_sequence_length = ds["min_sequence_length"]
    work_dir = ds["work_dir"]
    dataset_tokenized = ds["paths"]
    include_instance_metadata = ds["include_instance_metadata"]

    if "vsl_curriculum" not in ds:
        raise ValueError("Only VSL Curriculum is implemented")

    vsl = ds["vsl_curriculum"]
    curriculum = vsl["name"]
    num_cycles = vsl["num_cycles"]
    balanced = vsl["balanced"]

    # ---- data_loader ----
    dl = config["data_loader"]
    global_batch_size = dl["global_batch_size"]
    seed = dl["seed"]
    num_workers = dl["num_workers"]
    prefetch_factor = dl["prefetch_factor"]

    # ---- trainer ----
    tr = config["trainer"]
    rank_microbatch_size = tr["rank_microbatch_size"]
    save_overwrite = tr["save_overwrite"]

    # ---- callbacks ----
    cbs = tr["callbacks"]
    scheduler = cbs["lr_scheduler"]
    grad_clipper = cbs["grad_clipper"]
    checkpointer = cbs["checkpointer"]
    wandb = cbs["wandb"]
    downstream_evaluator = cbs["downstream_evaluator"]

    # max_duration is specified as {"value": 1, "unit": "epochs"} (or "steps")
    md = tr["max_duration"]
    if md["unit"] == "epochs":
        max_duration = Duration.epochs(md["value"])
    elif md["unit"] == "steps":
        max_duration = Duration.steps(md["value"])
    else:
        raise ValueError(f"Unsupported max_duration unit: {md['unit']}")

    save_folder = Path(tr["save_folder"], f"{model_name}_{peak_lr}_{global_batch_size}_{weight_decay}_{max_duration.value}")
    save_folder.mkdir(parents=True, exist_ok=True)

    match model_name:
        case "olmo2_170M":
            build_config = TransformerConfig.olmo2_170M
        case "olmo2_190M":
            build_config = TransformerConfig.olmo2_190M
        case "olmo2_600M":
            build_config = TransformerConfig.olmo2_600M
        case "olmo2_1B":
            build_config = TransformerConfig.olmo2_1B
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    model_config = build_config(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        fused_ops=False,
        use_flash=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, 
            param_dtype=DType.bfloat16, 
            reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=peak_lr,
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.weight"], 
                opts=dict(weight_decay=weight_decay)
            ) # Daniela, need to double check this.
        ],
    )

    dataset_config = NumpyDatasetConfig.glob(
        *dataset_tokenized,  # can be globs
        name=dataset_name,
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        vsl_curriculum=VSLCurriculumConfig(name=curriculum, num_cycles=num_cycles, balanced=balanced),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        include_instance_metadata=include_instance_metadata,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            rank_microbatch_size=rank_microbatch_size,
            save_overwrite=save_overwrite,
            metrics_collect_interval=tr["metrics_collect_interval"],
            cancel_check_interval=tr["cancel_check_interval"],
            max_duration=max_duration,
            load_key_mapping={
                # For backwards compatibility when loading older checkpoints.
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=scheduler["warmup_steps"])))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=grad_clipper["max_grad_norm"]))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=checkpointer["save_interval"],
                ephemeral_save_interval=checkpointer["ephemeral_save_interval"],
                save_async=checkpointer["save_async"],
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=str(save_folder),
                cancel_check_interval=wandb["cancel_check_interval"],
                enabled=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=downstream_evaluator["tasks"],
                tokenizer=tokenizer_config,
                eval_interval=downstream_evaluator["eval_interval"],
            ),
        )
        # .with_callback(
        #     "batch_metrics",
        #     BatchMetricCallback()
        # )
    ) 
    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    )

def get_dataset_dataloader_for_evals(config_filepath: str, epoch: int=1):
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"

    with open(config_filepath, "r") as f:
        config_dict = json.load(f)
    
    config = build_config(config_dict)
    print(config, flush=True)

    # Set RNG states on all devices.
    seed_all(config.init_seed)
    set_random_seeds(config.init_seed)

    device = get_default_device()

    world_mesh = config.model.build_mesh(device=device)

    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, collator=KASDataCollator(pad_token_id=dataset.pad_token_id, rank_batch_size=config.trainer.rank_microbatch_size), mesh=world_mesh)
    
    data_loader.reshuffle(epoch)

    return dataset, data_loader

def main(config_filepath: str):
    with open(config_filepath, "r") as f:
        config_dict = json.load(f)
    
    config = build_config(config_dict)
    print(config, flush=True)

    # Set RNG states on all devices.
    seed_all(config.init_seed)
    set_random_seeds(config.init_seed)

    device = get_default_device()

    # Build the world mesh, if needed.
    world_mesh = config.model.build_mesh(device=device)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )

    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, collator=KASDataCollator(pad_token_id=dataset.pad_token_id, rank_batch_size=config.trainer.rank_microbatch_size), mesh=world_mesh)

    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train
    trainer.fit()


if __name__ == "__main__":
    config_filepath = sys.argv[1]   # e.g., kas_config.json
    prepare_training_environment()
    try:
        main(config_filepath)
    finally:
        teardown_training_environment()
