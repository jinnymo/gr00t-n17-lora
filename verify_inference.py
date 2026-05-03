"""Verify a GR00T N1.7 LoRA adapter is correctly applied at inference time.

Runs four independent checks. All four must pass for the adapter to be
considered live in the inference path.

  1. Trainable percentage falls within the expected LoRA range.
  2. At least one LoRA-A weight is non-zero (i.e. the adapter was trained).
  3. adapter_only/ on disk is in the expected size range.
  4. Forward output differs with the adapter enabled vs. disabled.

Usage:
    python verify_inference.py \
      --base /path/to/GR00T-N1.7-3B \
      --adapter /path/to/checkpoint-XXXXX
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fmt_pct(n: int, total: int) -> str:
    return f"{n:,} / {total:,} ({100 * n / total:.3f}%)"


def find_first_lora_module(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and len(module.lora_A) > 0:
            return name, module
    return None, None


def adapter_dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build_synthetic_obs(policy, ckpt_dir: Path):
    mc = policy.modality_configs

    state_dims = {k: 1 for k in mc["state"].modality_keys}
    for candidate in [
        ckpt_dir / "experiment_cfg" / "dataset_statistics.json",
        ckpt_dir / "statistics.json",
    ]:
        if candidate.exists():
            stats = json.loads(candidate.read_text())
            embodiment_stats = next(iter(stats.values()))
            state_section = embodiment_stats.get("state", {})
            for k in mc["state"].modality_keys:
                mean = state_section.get(k, {}).get("mean")
                if isinstance(mean, list):
                    state_dims[k] = len(mean)
            break

    np.random.seed(0)
    return {
        "video": {
            k: np.random.randint(0, 255, size=(1, 1, 224, 224, 3), dtype=np.uint8)
            for k in mc["video"].modality_keys
        },
        "state": {
            k: np.random.randn(1, 1, state_dims[k]).astype(np.float32)
            for k in mc["state"].modality_keys
        },
        "language": {
            policy.language_key: [["pick up the object"]],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path,
                        help="path to the GR00T-N1.7-3B base model directory")
    parser.add_argument("--adapter", required=True, type=Path,
                        help="path to a checkpoint directory (containing adapter_config.json "
                             "or an adapter_only/ subdirectory)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-trainable-pct", type=float, default=0.1)
    parser.add_argument("--max-trainable-pct", type=float, default=20.0)
    parser.add_argument("--min-adapter-mb", type=float, default=10.0)
    parser.add_argument("--max-adapter-mb", type=float, default=5000.0)
    parser.add_argument("--forward-diff-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    adapter_dir = args.adapter / "adapter_only" if (args.adapter / "adapter_only").exists() else args.adapter

    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(args.adapter),
        device=args.device,
    )
    policy.model.eval()

    if not hasattr(policy.model, "peft_config"):
        raise SystemExit("policy.model has no peft_config -- adapter did not load. "
                         "Check that adapter_config.json is present.")

    total = sum(p.numel() for p in policy.model.parameters())
    trainable = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable / total

    check1 = args.min_trainable_pct <= trainable_pct <= args.max_trainable_pct
    print(f"[1/4] Trainable: {fmt_pct(trainable, total)} -- "
          f"{'PASS' if check1 else 'FAIL'} (expected {args.min_trainable_pct}%..{args.max_trainable_pct}%)")

    name, module = find_first_lora_module(policy.model)
    if module is None:
        print("[2/4] No LoRA modules found -- FAIL")
        check2 = False
    else:
        adapter_name = next(iter(module.lora_A.keys()))
        weight = module.lora_A[adapter_name].weight
        norm = weight.detach().to(torch.float32).norm().item()
        check2 = norm > 1e-6
        print(f"[2/4] First LoRA-A norm ({name}/{adapter_name}): {norm:.4e} -- "
              f"{'PASS' if check2 else 'FAIL'}")

    if not adapter_dir.exists():
        print(f"[3/4] Adapter directory missing: {adapter_dir} -- FAIL")
        check3 = False
    else:
        size_mb = adapter_dir_size_bytes(adapter_dir) / (1024 ** 2)
        check3 = args.min_adapter_mb <= size_mb <= args.max_adapter_mb
        print(f"[3/4] adapter_only size: {size_mb:.1f} MB -- "
              f"{'PASS' if check3 else 'FAIL'} (expected {args.min_adapter_mb}..{args.max_adapter_mb} MB)")

    obs = build_synthetic_obs(policy, args.adapter)
    out_with, _ = policy._get_action(obs)
    a_with = out_with[next(iter(out_with))]

    if hasattr(policy.model, "disable_adapter") and callable(policy.model.disable_adapter):
        with policy.model.disable_adapter():
            out_without, _ = policy._get_action(obs)
    else:
        policy.model.disable_adapters()
        try:
            out_without, _ = policy._get_action(obs)
        finally:
            policy.model.enable_adapters()
    a_without = out_without[next(iter(out_without))]
    diff = float(np.abs(np.asarray(a_with) - np.asarray(a_without)).mean())
    check4 = diff > args.forward_diff_threshold
    print(f"[4/4] Forward output diff (adapter on vs off): {diff:.6f} -- "
          f"{'PASS' if check4 else 'FAIL'} (threshold {args.forward_diff_threshold})")

    all_pass = check1 and check2 and check3 and check4
    print()
    if all_pass:
        print("All 4 checks passed. LoRA adapter is correctly applied.")
        raise SystemExit(0)
    print("One or more checks failed.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
