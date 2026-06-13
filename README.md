# gr00t-n17-lora

LoRA fine-tuning for NVIDIA GR00T N1.7 (3B params) on a single 24 GB GPU.

NVIDIA's N1.7 release does not ship LoRA support. Commit `4e62473` on
the n1.6 branch removed it, and main / N1.7 do not bring it back. This
repo restores LoRA on N1.7 by monkey-patching three hooks in
Isaac-GR00T, without modifying the upstream source tree.

> Status: pre-release. A working LoRA adapter is published at
> [`dongyoonkim/grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1),
> and the matching 90-episode SO-ARM101 wrist-only training dataset at
> [`dongyoonkim/so101-eraser-90ep-wrist`](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist).
> The adapter is wired into the Quick Start below.

## Documentation

- [Debugging journey](docs/debugging_journey.md) — how LoRA was restored on N1.7.
- [Results](docs/results.md) — open-loop MAE and real-robot numbers.
- [Verification](docs/verification.md) — what the four checks confirm.

## Verified environment

This wrapper has been tested end-to-end against:

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 LTS (kernel 6.17) |
| GPU | NVIDIA RTX 3090 Ti, 24 GB |
| NVIDIA driver | 590.48.01 |
| CUDA (torch build) | 12.6 |
| cuDNN | 9.5.1 |
| Python | 3.10.20 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1 |
| transformers | 4.57.3 |
| peft | 0.17.1 |
| accelerate | 1.13.0 |
| diffusers | 0.35.1 |
| safetensors | 0.7.0 |
| tokenizers | 0.22.2 |
| numpy | 1.26.4 |
| pyarrow | 24.0.0 |
| tyro | 0.9.17 |
| bitsandbytes | 0.49.2 (optional) |
| Isaac-GR00T | commit pinning torch 2.7.1 / flash-attn 2.7.4.post1 |

Other versions may work but have not been verified.

## Install

This wrapper imports `gr00t.*` from NVIDIA's
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), which must be
installed first. Pick one of the two paths below.

### Option A: reuse an existing Isaac-GR00T conda env (recommended)

If you already have Isaac-GR00T installed and working, install this
repo's extras into the same environment:

```bash
git clone https://github.com/jinnymo/gr00t-n17-lora
cd gr00t-n17-lora
pip install -r requirements.txt
```

### Option B: create a fresh conda env from scratch

```bash
# 1. Create the conda env with pinned dependencies.
git clone https://github.com/jinnymo/gr00t-n17-lora
cd gr00t-n17-lora
conda env create -f environment.yml
conda activate grootn17-lora

# 2. Install Isaac-GR00T into the same env.
git clone https://github.com/NVIDIA/Isaac-GR00T ../Isaac-GR00T
pip install -e ../Isaac-GR00T

# 3. Install the flash-attn pre-built wheel that Isaac-GR00T pins
#    (Isaac-GR00T's pyproject.toml lists the exact URL for torch 2.7 / cu126).

# 4. Install this repo's wrapper extras.
pip install -r requirements.txt
```

## 5-minute Quick Start (verification)

After install, confirm the LoRA path is live by running four checks
against a checkpoint:

```bash
# Download GR00T N1.7 base (~6.5 GB; skip if you already have it).
huggingface-cli download nvidia/GR00T-N1.7-3B \
  --local-dir models/GR00T-N1.7-3B

# Download the public LoRA adapter (~2.2 GB).
huggingface-cli download dongyoonkim/grootn17-lora-so101-eraser-tier1 \
  --local-dir checkpoints/eraser-tier1

# Run the four checks.
python verify_inference.py \
  --base models/GR00T-N1.7-3B \
  --adapter checkpoints/eraser-tier1
```

Expected output:

```
[1/4] Trainable: ... (~14%) -- PASS
[2/4] First LoRA-A norm: ... -- PASS
[3/4] adapter_only size: ... MB -- PASS
[4/4] Forward output diff (adapter on vs off): ... -- PASS

All 4 checks passed. LoRA adapter is correctly applied.
```

## Files

- `launch_finetune_grootn17_lora.py` — wrapper around Isaac-GR00T's
  `launch_finetune.py` that injects LoRA via PEFT.
- `verify_inference.py` — four checks confirming the adapter applies.
- `eval_open_loop.py` — open-loop MAE across multiple checkpoints with
  a single base model load (adapters are swapped).
- `examples/so101_wrist_only_config.py` — modality config for SO-ARM101
  with a single wrist camera.
- `docs/debugging_journey.md` — how LoRA was restored on N1.7.
- `docs/results.md` — open-loop MAE and real-robot numbers.
- `docs/verification.md` — what the four checks confirm.

## Related

- LoRA adapter trained with this wrapper: [huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1)
- Training dataset (90 ep, SO-ARM101 wrist-only): [huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist)
- LeRobot v3.0 ↔ v2.1 format converter: [github.com/jinnymo/lerobot-v3-v2-converter](https://github.com/jinnymo/lerobot-v3-v2-converter)

## License

Apache License 2.0. See [LICENSE](./LICENSE).
