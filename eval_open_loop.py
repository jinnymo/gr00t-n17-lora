"""Open-loop MAE evaluation for GR00T N1.7 LoRA checkpoints.

Loads the base model once and swaps adapters across checkpoints to amortize
the 6.5 GB load. Reuses gr00t.eval.open_loop_eval.evaluate_single_trajectory.

Usage:
    python eval_open_loop.py \
      --ckpt-base /path/to/output_dir \
      --dataset-path /path/to/lerobot_dataset_v2.1 \
      --steps 3000 6000 9000 12000 15000 \
      --traj-ids 0 1 2 \
      --out-json results.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import evaluate_single_trajectory
from gr00t.policy.gr00t_policy import Gr00tPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-base", required=True, type=Path,
                        help="parent directory containing checkpoint-XXXX subdirs")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                        help="checkpoint steps to evaluate")
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="frames per trajectory")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    first_step = args.steps[0]
    first_ckpt = args.ckpt_base / f"checkpoint-{first_step}"
    if not first_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {first_ckpt}")

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(first_ckpt),
        device="cuda",
    )

    if not hasattr(policy.model, "peft_config"):
        raise RuntimeError(
            f"policy.model has no peft_config (got {type(policy.model).__name__}). "
            "Verify that adapter_config.json is present in the checkpoint directory."
        )

    initial_adapter = next(iter(policy.model.peft_config.keys()))

    for step in args.steps:
        if step == first_step:
            continue
        ckpt = args.ckpt_base / f"checkpoint-{step}"
        adapter_dir = ckpt / "adapter_only" if (ckpt / "adapter_only").exists() else ckpt
        if not adapter_dir.exists():
            print(f"[skip] {adapter_dir} missing")
            continue
        policy.model.load_adapter(str(adapter_dir), adapter_name=f"ckpt{step}")

    modality = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=str(args.dataset_path),
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    results = {}
    for step in args.steps:
        adapter_name = initial_adapter if step == first_step else f"ckpt{step}"
        if adapter_name not in policy.model.peft_config:
            continue

        policy.model.set_adapter(adapter_name)
        per_traj = []
        for traj_id in args.traj_ids:
            if traj_id >= len(dataset):
                continue
            plot_path = (
                str(args.plot_dir / f"ckpt{step}_traj{traj_id}.jpeg")
                if args.plot_dir else None
            )
            mse, mae = evaluate_single_trajectory(
                policy=policy,
                loader=dataset,
                traj_id=traj_id,
                embodiment_tag=embodiment_tag,
                modality_keys=None,
                steps=args.eval_steps,
                action_horizon=args.action_horizon,
                save_plot_path=plot_path,
            )
            per_traj.append({"traj_id": traj_id, "mae": float(mae), "mse": float(mse)})

        avg_mae = float(np.mean([t["mae"] for t in per_traj])) if per_traj else float("nan")
        avg_mse = float(np.mean([t["mse"] for t in per_traj])) if per_traj else float("nan")
        results[step] = {"avg_mae": avg_mae, "avg_mse": avg_mse, "per_traj": per_traj}

    print()
    print(f"{'ckpt':>8} | {'Avg MAE (deg)':>14} | {'Avg MSE':>10}")
    print("-" * 40)
    for step, r in sorted(results.items()):
        print(f"{step:>8} | {r['avg_mae']:>14.3f} | {r['avg_mse']:>10.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "ckpt_base": str(args.ckpt_base),
        "dataset_path": str(args.dataset_path),
        "traj_ids": args.traj_ids,
        "eval_steps": args.eval_steps,
        "action_horizon": args.action_horizon,
        "results": {str(k): v for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
