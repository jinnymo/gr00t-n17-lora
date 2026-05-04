# Results

All numbers below come from a single SO-ARM101 setup with a single
wrist camera (Innomaker U20CAM-1080P, 640 × 480 @ 30 fps), training
on the
[`dongyoonkim/so101-eraser-90ep-wrist`](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist)
dataset (90 demonstrations, task `"Place the white eraser on the red square"`,
all successful trajectories, no recovery behaviour).

## 1. Open-loop joint-error (MAE)

Mean absolute joint error (degrees) across three held-back
trajectories from the same dataset, evaluated at five training
checkpoints. Lower is better.

| Step | Avg MAE (°) | Avg MSE | Per-traj MAE (0 / 1 / 2) |
|---|---|---|---|
| 3 000  | 3.846 | 36.51 | 3.955 / 4.033 / 3.550 |
| 6 000  | 2.803 | 25.57 | 2.908 / 2.717 / 2.784 |
| 9 000  | 2.347 | 20.15 | 2.325 / 2.316 / 2.398 |
| 12 000 | 1.860 | 12.73 | 2.008 / 1.762 / 1.811 |
| **15 000** | **1.661** | **10.85** | 1.563 / 1.755 / 1.666 |

- Monotonic decrease across all five checkpoints; no overfit
  inflection inside the 15 000 steps.
- Step 15 000 is the final checkpoint and is the one published as
  [`grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1).
- Per-trajectory standard deviation at step 15 000 is ~0.10 °, so
  the mean is stable across trajectories.

### Train-loss / open-loop MAE alignment

For the same five checkpoints:

| Step | Mean train loss (window) | Open-loop MAE | `loss × 100 / MAE` |
|---|---|---|---|
| 3 000  | 0.071 | 3.846 | 1.85 |
| 6 000  | 0.052 | 2.803 | 1.86 |
| 9 000  | 0.040 | 2.347 | 1.71 |
| 12 000 | 0.030 | 1.860 | 1.61 |
| 15 000 | 0.026 | 1.661 | 1.57 |

The ratio is consistent (1.57 – 1.86), i.e. train loss tracks
held-out MAE roughly linearly. This is the property that an
earlier attention-only LoRA configuration on the same wrapper did
*not* have.

## 2. Comparison against other configurations on the same dataset

| Setup | Train method | Best ckpt MAE (°) | Adapter size | Real-robot success |
|---|---|---|---|---|
| GR00T N1.7 full fine-tune | backbone + action head, all params | **1.297** | ~15 GB (no adapter) | **46 %** (6 / 13) |
| GR00T N1.7 + LoRA `r=32`, attention-only (no `modules_to_save`, no MLP) | LoRA only | 6.133 | ~90 MB | not evaluated on real |
| SmolVLA + LoRA `r=64` (LeRobot native PEFT) | LoRA | 2.347 | ~150 MB | 35.7 % (5 / 14) |
| **GR00T N1.7 + LoRA `r=32` + Tier 1** (this wrapper, adapter-only release) | LoRA + 6 `modules_to_save` + MLP targets | **1.661** | **~2.2 GB** | **~40 %** |

"Tier 1" refers to the configuration also used in this release:
attention + MLP LoRA targets, plus six fully-trainable
`modules_to_save` (state encoder, action encoder, action decoder,
position embedding, vlln, vl_self_attention).

Two takeaways:

- Full fine-tune is still ~28 % better in open-loop MAE (1.30 vs
  1.66 °), but the LoRA Tier 1 adapter is **roughly 1/7** the
  size on disk (~2.2 GB vs ~15 GB).
- Real-robot success rate sits at ~40 %, against 46 % for the
  full fine-tune of the same base on the same dataset.

## 3. Real-robot evaluation

Closed-loop on SO-ARM101 with the `gr00t.eval.run_gr00t_server`
service (action horizon 8, no sub-frame interpolation,
`max_deg_per_step` unlimited). Task instruction at inference time:
`"Place the white eraser on the red square"`.

| Position bucket | Trials | Success |
|---|---|---|
| Easy (eraser on / near the red square, wrist-camera view square-aligned) | 3 | **3 / 3 (100 %)** |
| Hard (eraser at edges or outside the centred 10 cm radius) | several | **0** |
| **Combined** | — | **~40 %** (lands in the 30 – 60 % "partial success" band) |

In the easy-position trials the policy executes the full
sequence — approach, grip, lift, transport, release — without
shake or visible delay. In the hard-position trials the policy
reliably grips and lifts but stalls before transporting.

### Why it stalls — Phase D (transport) OOD trigger failure

Two unattended 30-second trials were recorded with the same
checkpoint, action chunks streamed continuously to disk, and then
inspected by joint trajectory. The behaviour decomposes into:

| Phase | Status |
|---|---|
| A — home → approach | learned, deterministic |
| B — grip close | learned |
| C — lift | learned |
| D — transport | **fragile, fires only when the observation falls inside a narrow band** |
| E — place + grip open | learned (observed firing in run 2) |
| F — re-grip after drop | learned (observed firing in run 2) — not present in the full-fine-tune baseline |

Run 2, in particular, exhibits a complete drop-and-recover cycle:
the gripper opens to ~33 ° over the red square (a clean release),
the arm descends, the gripper closes again, the arm lifts, and
then stalls in the same lifted pose as run 1.

This is **not** a mode-collapse failure. The chunks across phases
are distinct — gripper goes 0.72 ° → 33 ° → 0.72 °, shoulder lift
spans roughly -100 ° to +20 °. The failure is specifically at the
**transport (Phase D) trigger**: when the observation is outside
the trained distribution, Phase D never fires and the policy
loops the same lifted-pose chunk.

Inference latency was measured at p50 105.7 ms with no measurable
gap between consecutive chunks at action horizon 8, so latency is
not the cause of the stall.

## 4. Why 90 episodes was not enough

The same 90-episode dataset caps three different training methods
(full fine-tune, SmolVLA + LoRA, this wrapper) at ~36 – 46 %. The
~10 % spread between methods is small relative to the gap to a
hypothetical 70 %+ system. Most of the remaining gap is therefore
**dataset-bound, not adapter-bound**.

Concretely:

- **Volume.** 90 episodes is on the lower end for LoRA
  fine-tuning on a task that requires recovery behaviour. The
  NVIDIA SO-101 + GR00T N1.5 example achieves a high success rate
  on a far simpler block-stacking task at ~50 episodes; tasks
  involving variable starting positions and recovery typically
  want 150 – 300.
- **Trajectory variety.** All 90 episodes are successful
  demonstrations. There are zero failure-and-recover trajectories
  in the dataset, which is why Phase F (re-grip) fires
  inconsistently in the deployed policy.
- **Starting-pose distribution.** Eraser starting positions are
  concentrated within a ~10 cm radius of the red square, with the
  wrist camera roughly aligned to the square. Hard positions
  (corners, off-square) are sparse in the training distribution,
  so the Phase D transport trigger is fragile precisely there.
- **Base-model prior.** LoRA inherits the prior of the base
  GR00T N1.7 (3.7 B). The 14 % trainable share of this Tier 1
  adapter is enough to retarget the action / state spaces to a
  6-DoF SO-ARM101, but cannot shift the base prior far enough
  to compensate for a narrow training distribution.

Trying to push real-robot success past ~46 % on this base
therefore requires more, more varied, recovery-bearing data
rather than more LoRA capacity.

## 5. Headline numbers

- Best open-loop MAE: **1.661 °** at step 15 000 (Tier 1, `r=32`).
- Adapter on disk: **~2.2 GB**, vs ~15 GB for a full fine-tune.
- Real-robot success: **~40 %**, against 46 % for a full
  fine-tune of the same base on the same data — i.e. the LoRA
  adapter recovers ~87 % of the full fine-tune's real-world
  success at ~1/7 the artifact size.
