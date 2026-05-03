"""LoRA fine-tuning wrapper for NVIDIA GR00T N1.7.

Restores LoRA support that was removed in the n1.6 branch (commit 4e62473)
and is absent from main / N1.7. Works without modifying the Isaac-GR00T
source tree by monkey-patching three hooks:

  1. Gr00tN1d7Pipeline._create_model -- inject get_peft_model after build
  2. Gr00tTrainer.save_model -- also write adapter_only/ on each checkpoint
  3. forward(**kwargs) bridge -- HF Trainer keyword-call into GR00T's dict signature

Usage:
    python launch_finetune_grootn17_lora.py \
      --dataset-path /path/to/your/lerobot_dataset_v2.1 \
      --base-model-path /path/to/GR00T-N1.7-3B \
      --output-dir /path/to/checkpoints/run \
      --modality-config-path examples/so101_wrist_only_config.py \
      --embodiment-tag NEW_EMBODIMENT \
      --max-steps 15000 --save-steps 3000 \
      --global-batch-size 16 --learning-rate 1e-4 \
      --lora_rank 32 --lora_alpha 64 \
      --lora_include_mlp --lora_modules_to_save_action_head \
      --no-tune_diffusion_model \
      --use-wandb --wandb-project your_project
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import bitsandbytes  # noqa: F401
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

import torch
import tyro
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback  # noqa: F401  -- kept for downstream callbacks

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.model.base.model_pipeline import BasicPipeline
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline


@dataclass
class Gr00tLoraConfig(FinetuneConfig):
    """FinetuneConfig extended with LoRA hyperparameters.

    lora_rank=0 disables LoRA (falls through to full fine-tuning).
    """
    lora_rank: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_action_head_only: bool = False

    lora_include_mlp: bool = False
    lora_modules_to_save_action_head: bool = False


_LOG_PREFIX = "[grootn17-lora]"


def get_lora_model(model, rank=16, lora_alpha=32, lora_dropout=0.0,
                   action_head_only=False, include_mlp=False,
                   modules_to_save_action_head=False):
    attn_patterns = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "to_q", "to_v", "to_k", "to_out.0",
    ]
    mlp_patterns = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "ff.net.0.proj", "ff.net.2",
    ] if include_mlp else []
    name_patterns = attn_patterns + mlp_patterns

    target_modules = []
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in name_patterns):
                target_modules.append(name)

    if not target_modules:
        raise ValueError(
            f"No target_modules matched. include_mlp={include_mlp}, "
            f"name_patterns={name_patterns}"
        )

    modules_to_save = None
    if modules_to_save_action_head:
        modules_to_save = [
            "state_encoder",
            "action_encoder",
            "action_decoder",
            "position_embedding",
            "vlln",
            "vl_self_attention",
        ]

    print(f"{_LOG_PREFIX} target_modules: {len(target_modules)} layers, include_mlp={include_mlp}")
    print(f"{_LOG_PREFIX} modules_to_save: {modules_to_save}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=lora_dropout,
        bias="none",
        # task_type=None keeps the generic PeftModel signature; FEATURE_EXTRACTION
        # and CAUSAL_LM force NLP-style (input_ids) calls that don't match GR00T.
        task_type=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Bridge HF Trainer's keyword-call into GR00T's positional dict signature.
    gr00t_model = model.base_model.model
    original_forward = gr00t_model.forward

    def patched_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    gr00t_model.forward = patched_forward
    return model


_ORIGINAL_CREATE_MODEL = BasicPipeline._create_model
_ORIGINAL_CREATE_MODEL_N1D7 = Gr00tN1d7Pipeline._create_model
_LORA_RUNTIME_CFG: dict = {}


def _wrap_model_with_lora(model):
    cfg = _LORA_RUNTIME_CFG
    if cfg.get("rank", 0) <= 0:
        return model
    return get_lora_model(
        model,
        rank=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        action_head_only=cfg.get("action_head_only", False),
        include_mlp=cfg.get("include_mlp", False),
        modules_to_save_action_head=cfg.get("modules_to_save_action_head", False),
    )


def _create_model_with_lora(self):
    model = _ORIGINAL_CREATE_MODEL(self)
    return _wrap_model_with_lora(model)


def _create_model_n1d7_with_lora(self):
    model = _ORIGINAL_CREATE_MODEL_N1D7(self)
    return _wrap_model_with_lora(model)


BasicPipeline._create_model = _create_model_with_lora
Gr00tN1d7Pipeline._create_model = _create_model_n1d7_with_lora


_ORIG_SAVE_MODEL = Gr00tTrainer.save_model


def _save_model_with_lora(self, output_dir=None, _internal_call=False):
    _ORIG_SAVE_MODEL(self, output_dir, _internal_call)

    if output_dir is None:
        output_dir = self.args.output_dir

    # HF Trainer.train() reassigns self.model = unwrap_model(self.model, recursive=True),
    # which strips PEFT. self.model_wrapped retains the original PeftModel.
    candidates = [
        getattr(self, "model_wrapped", None),
        self.model,
    ]
    peft_model = None
    for m in candidates:
        if m is None:
            continue
        underlying = getattr(m, "module", m)
        if hasattr(underlying, "peft_config"):
            peft_model = underlying
            break

    if peft_model is None:
        return

    adapter_dir = Path(output_dir) / "adapter_only"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))


Gr00tTrainer.save_model = _save_model_with_lora


def load_modality_config(modality_config_path: str):
    import importlib

    path = Path(modality_config_path)
    if not (path.exists() and path.suffix == ".py"):
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)


def main():
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(Gr00tLoraConfig, description=__doc__)

    global _LORA_RUNTIME_CFG
    _LORA_RUNTIME_CFG = {
        "rank": ft_config.lora_rank,
        "alpha": ft_config.lora_alpha,
        "dropout": ft_config.lora_dropout,
        "action_head_only": ft_config.lora_action_head_only,
        "include_mlp": ft_config.lora_include_mlp,
        "modules_to_save_action_head": ft_config.lora_modules_to_save_action_head,
    }

    if ft_config.lora_rank > 0:
        print(f"{_LOG_PREFIX} LoRA enabled (rank={ft_config.lora_rank}, alpha={ft_config.lora_alpha})")
    else:
        print(f"{_LOG_PREFIX} LoRA disabled (lora_rank=0)")

    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    config.training.gradient_checkpointing = True
    config.training.optim = "paged_adamw_8bit" if _HAS_BNB else "adamw_torch"

    run(config)


if __name__ == "__main__":
    main()
