"""
Official training script for OLMo-2-0325-32B, meant to be launched with torchrun.
"""
from __future__ import annotations

import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import (
    WSD,
    ConstantScheduler,
    ConstantWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import LogFilterType, seed_all
from omegaconf import OmegaConf


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_eval_dataset(
    patterns: str | list[str],
    *,
    tokenizer_cfg,
    seq_len: int,
    work_dir: str,
) -> tuple[NumpyDatasetConfig, list[dict]]:
    """
    Collect one or many *.npy glob patterns into one NumpyDatasetConfig for multiple folder eval
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    merged_cfg = None
    metadata = []

    for pat in patterns:
        files = [f for f in glob.iglob(pat) if Path(f).is_file()]
        print(f"files for pattern: {pat} {len(files)}")
        if not files:
            continue

        cfg = NumpyPaddedFSLDatasetConfig.glob(
            pat,
            tokenizer=tokenizer_cfg,
            sequence_length=seq_len,
            work_dir=work_dir,
        )

        label = Path(pat).parent.name
        metadata.extend([{"label": label}] * len(files))

        if merged_cfg is None:
            merged_cfg = cfg
        else:
            merged_cfg.paths.extend(cfg.paths)

    merged_cfg.metadata = metadata
    return merged_cfg, metadata


def build_config(config_yaml: dict, overrides: list[str]) -> ExperimentConfig:
    tokenizer_config = get_tokenizer_config(config_yaml["tokenizer_name"])()
    model_config = get_model_config(config_yaml["model_name"])(
        vocab_size=tokenizer_config.padded_vocab_size(),
        use_flash=config_yaml["use_flash_attention_2"],
    )

    dataset_config = NumpyFSLDatasetConfig.glob(
        config_yaml["data_root"],
        tokenizer=tokenizer_config,
        sequence_length=config_yaml["sequence_length"],
        max_target_sequence_length=max(8192, config_yaml["sequence_length"]),
        work_dir=config_yaml["work_dir"],
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=config_yaml["gradient_accumulation_steps"]
        * config_yaml["per_device_batch_size"]
        * get_world_size()
        * config_yaml["sequence_length"],
        seed=config_yaml["seed"],
        num_workers=config_yaml["num_workers"],
    )
    scheduler_type = config_yaml["scheduler"]["type"]
    if scheduler_type == "ConstantWithWarmup":
        scheduler = ConstantWithWarmup(
            warmup_steps=config_yaml["scheduler"].get("warmup_steps"),
            warmup_fraction=(config_yaml["scheduler"].get("warmup_fraction")),
            warmup_min_lr=config_yaml["scheduler"]["warmup_min_lr"],
        )
    elif scheduler_type == "ConstantScheduler":
        scheduler = ConstantScheduler()
    elif scheduler_type == "WSD":
        scheduler = WSD(
            warmup=config_yaml["scheduler"].get("warmup_steps"),
            warmup_fraction=(config_yaml["scheduler"].get("warmup_fraction")),
            warmup_delay=config_yaml["scheduler"].get("warmup_delay"),
            decay_steps=config_yaml["scheduler"].get("decay_steps"),
            decay_fraction=config_yaml["scheduler"].get("decay_fraction"),
            warmup_min_lr=config_yaml["scheduler"].get("warmup_min_lr"),
            decay_min_lr=config_yaml["scheduler"].get("decay_min_lr"),
        )
    else:
        raise ValueError(f"scheduler_type: {scheduler_type}")
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=config_yaml["per_device_batch_size"] * config_yaml["sequence_length"],
        max_sequence_length=config_yaml["sequence_length"],
        optim=SkipStepAdamWConfig(
            lr=config_yaml["learning_rate"],
            weight_decay=config_yaml["weight_decay"],
            betas=config_yaml["betas"],
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})],
            compile=config_yaml["optimizer_compile"],
        ),
        scheduler=scheduler,
        compile_model=config_yaml["model_compile"],
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=get_world_size(),  # NOTE: tune this
        ),
        ac_config=TransformerActivationCheckpointingConfig(TransformerActivationCheckpointingMode.full),
        z_loss_multiplier=config_yaml["z_loss_multiplier"],
        max_grad_norm=config_yaml["max_grad_norm"],
    )

    # If you have 1024 GPUs, you can run slightly faster with a different config.
    if get_world_size() >= 1024:
        train_module_config.rank_microbatch_size //= 2
        train_module_config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    eval_cfg, _ = build_eval_dataset(
        config_yaml["data_root_val"],
        tokenizer_cfg=tokenizer_config,
        seq_len=config_yaml["sequence_length"],
        work_dir=config_yaml["work_dir"],
    )
    max_duration_tokens = int(config_yaml["max_duration"].get("tokens", -1))
    max_duration_epochs = int(config_yaml["max_duration"].get("epochs", 1))
    max_duration_steps = int(config_yaml["max_duration"].get("steps", -1))
    if max_duration_tokens > 0:
        duration = Duration.tokens(max_duration_tokens)
    elif max_duration_steps > 0:
        duration = Duration.steps(max_duration_steps)
    else:
        duration = Duration.epochs(max_duration_epochs)
    trainer_config = (
        TrainerConfig(
            save_folder=f"{config_yaml['save_dir']}",
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=duration,
            hard_stop=Duration.steps(int(config_yaml["hard_stop"])) if int(config_yaml["hard_stop"]) > 0 else None,
            load_path=config_yaml["load_path"],
            load_strategy=LoadStrategy.if_available,
            restart_with_new_data=config_yaml["restart_with_new_data"],
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=config_yaml["save_interval"],
                ephemeral_save_interval=config_yaml["ephemeral_save_interval"],
                save_async=config_yaml["save_interval"],
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=config_yaml["run_name"],
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=config_yaml["run_name"],
                cancel_check_interval=10,
                enabled=True,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                device_peak_flops=int(
                    989e12
                )  # Need to provide this, otherwise the pre_train does not set it correctly
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=eval_cfg,
                eval_interval=config_yaml["evaluation"]["lm_eval"]["interval"],
                enabled=config_yaml["evaluation"]["lm_eval"]["enabled"],
                eval_max_steps=config_yaml["evaluation"]["lm_eval"]["max_steps"],
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    #             # MMLU for backwards compatibility
                    #             "mmlu_stem_mc_5shot",
                    #             "mmlu_humanities_mc_5shot",
                    #             "mmlu_social_sciences_mc_5shot",
                    #             "mmlu_other_mc_5shot",
                    #             # MMLU test
                    #             "mmlu_stem_mc_5shot_test",
                    #             "mmlu_humanities_mc_5shot_test",
                    #             "mmlu_social_sciences_mc_5shot_test",
                    #             "mmlu_other_mc_5shot_test",
                    #             ## Core 12 tasks for backwards compatibility
                    #             # "arc_challenge",
                    #             # "arc_easy",
                    #             # "basic_arithmetic",
                    #             # "boolq",
                    #             # "commonsense_qa",
                    #             # "copa",
                    #             # "hellaswag",
                    #             # "openbook_qa",
                    #             # "piqa",
                    #             # "sciq",
                    #             # "social_iqa",
                    #             # "winogrande",
                    #             ## Core 12 tasks 5-shot
                    #             # "arc_challenge_rc_5shot",
                    #             # "arc_easy_rc_5shot",
                    #             ## "basic_arithmetic_rc_5shot",  # doesn't exist
                    #             ## "boolq_rc_5shot",  # we don't like it
                    #             # "csqa_rc_5shot",
                    #             ## "copa_rc_5shot",  # doesn't exist
                    #             # "hellaswag_rc_5shot",
                    #             # "openbookqa_rc_5shot",
                    #             # "piqa_rc_5shot",
                    #             ## "sciq_rc_5shot",  # doesn't exist
                    #             # "socialiqa_rc_5shot",
                    #             # "winogrande_rc_5shot",
                    #             ## New in-loop evals
                    #             # "arc_challenge_val_rc_5shot",
                    #             # "arc_challenge_val_mc_5shot",
                    "arc_challenge_test_rc_5shot",
                    #             # "arc_challenge_test_mc_5shot",
                    #             # "arc_easy_val_rc_5shot",
                    #             # "arc_easy_val_mc_5shot",
                    "arc_easy_test_rc_5shot",
                    #             # "arc_easy_test_mc_5shot",
                    #             # "boolq_val_rc_5shot",
                    #             # "boolq_val_mc_5shot",
                    #             "csqa_val_rc_5shot",
                    #             # "csqa_val_mc_5shot",
                    #   "hellaswag_val_rc_5shot",
                    #             # "hellaswag_val_mc_5shot",
                    #             # "openbookqa_val_rc_5shot",
                    #             # "openbookqa_val_mc_5shot",
                    #             "openbookqa_test_rc_5shot",
                    #             # "openbookqa_test_mc_5shot",
                    "piqa_val_rc_5shot",
                    #             # "piqa_val_mc_5shot",
                    #             "socialiqa_val_rc_5shot",
                    #             # "socialiqa_val_mc_5shot",
                    #             # "winogrande_val_rc_5shot",
                    #             # "winogrande_val_mc_5shot",
                    # "mmlu_stem_val_rc_5shot",
                    #             # "mmlu_stem_val_mc_5shot",
                    #             # "mmlu_humanities_val_rc_5shot",
                    #             # "mmlu_humanities_val_mc_5shot",
                    #             # "mmlu_social_sciences_val_rc_5shot",
                    #             # "mmlu_social_sciences_val_mc_5shot",
                    #             # "mmlu_other_val_rc_5shot",
                    #             # "mmlu_other_val_mc_5shot",
                    # "mmlu-nl_stem_mc_5shot",
                    # "mmlu-nl_humanities_mc_5shot",
                    # "mmlu-nl_social_sciences_mc_5shot",
                    # "mmlu-nl_social_sciences_mc_5shot_test",
                    # "hellaswag-nl_rc_0shot",
                    # "mmlu-nl_stem_mc_5shot",
                    # "hellaswag-nl_val_mc_5shot",
                    # "mmlu-nl_humanities_mc_5shot",
                    "hellaswag-nl_rc_0shot",
                    "mmlu-nl_stem_mc_5shot",
                ],
                tokenizer=tokenizer_config,
                eval_interval=config_yaml["evaluation"]["downstream_tasks"]["interval"],
                enabled=config_yaml["evaluation"]["downstream_tasks"]["enabled"],
                eval_max_steps=config_yaml["evaluation"]["downstream_tasks"]["max_steps"],
            ),
        )
    )
    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(config_yaml: dict, overrides: list[str]):
    config = build_config(config_yaml, overrides)
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


def get_model_config(model_name: str):
    method_name = model_name.replace("-", "_")

    if not hasattr(TransformerConfig, method_name):
        raise ValueError(f"Model config '{model_name}' not found in TransformerConfig.")

    return getattr(TransformerConfig, method_name)


def get_tokenizer_config(tokenizer_name: str):
    method_name = tokenizer_name.replace("-", "_")

    if hasattr(TokenizerConfig, method_name):
        return getattr(TokenizerConfig, method_name)

    def fallback():
        try:
            return TokenizerConfig.from_hf(tokenizer_name)
        except Exception as e:
            raise ValueError(
                f"Tokenizer '{tokenizer_name}' not found in TokenizerConfig and failed to load from Hugging Face."
            ) from e

    return fallback


def load_config(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    return config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} config_yaml [OVERRIDES...]")
        sys.exit(1)

    config_name, *overrides = sys.argv[1:]

    prepare_training_environment(log_filter_type=LogFilterType.rank0_only)
    config_yaml = load_config(config_name)
    config_yaml["per_device_batch_size"] = int(config_yaml["per_device_batch_size"])
    config_yaml["gradient_accumulation_steps"] = int(config_yaml["gradient_accumulation_steps"])
    try:
        main(config_yaml, overrides=overrides)
    finally:
        teardown_training_environment()
