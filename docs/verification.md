# Verification: the four checks

[`verify_inference.py`](../verify_inference.py) loads the base
model and a candidate adapter and runs four independent checks
that, together, confirm a LoRA adapter is actually applied (rather
than silently falling back to a full fine-tune or to no
modification at all).

The numbers below were measured against
[`dongyoonkim/grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1)
on the pinned environment from the top-level README
(transformers 4.57.3 / peft 0.17.1 / torch 2.7.1+cu126 / RTX 3090 Ti).

## 1. Trainable-parameter share is in the LoRA range

```
Trainable: 528,793,728 / 3,721,568,512 = 14.21%
```

A correctly applied LoRA on this wrapper reports between roughly
0.3 % and 16 % trainable, depending on the configuration:

| Config | Trainable | Note |
|---|---|---|
| Attention-only LoRA, action head frozen | ~0.4 % | smallest viable adapter |
| Attention + MLP LoRA, action head frozen | ~0.7 % | |
| Attention + MLP LoRA + `modules_to_save` action head | **~14 – 16 %** | this adapter |
| No LoRA, default GR00T tune flags | ~16.8 % | action-head selective FT only — easy to mistake for LoRA |

The two ~15 % configurations look superficially similar by this
metric alone, which is exactly why this check is not sufficient
on its own — checks 2 – 4 disambiguate them.

> The training-time number from logs is ~15.5 %; the inference-time
> number is ~14.2 %. The difference comes from `requires_grad`
> bookkeeping on modules that PEFT marks as trainable during
> training but that are not flagged trainable when the model is
> reloaded from disk for inference. Both values are within the
> expected band; the inference-time number is the one this check
> reports.

## 2. The first LoRA-A layer has non-zero weights

```
First LoRA-A norm (backbone q_proj/default): 3.4491
```

PEFT initialises `lora_A` with Kaiming and `lora_B` with zero, so
freshly initialised adapters have a non-trivial `lora_A` norm but
produce zero output (because `B @ A = 0` until `B` learns). After
training, `lora_A` should remain non-zero — if it has collapsed to
zero, the adapter has been merged into the base or otherwise lost.

The check fails only when the norm is exactly zero or NaN. The
specific value (~3.4 here) is not load-bearing; the point is that
the layer exists and contains data.

## 3. `adapter_only/` size on disk is in the LoRA range

```
adapter_only size: 2210 MB (~2.2 GB)
```

The expected size depends on rank and on whether `modules_to_save`
is set. For this configuration:

- `r=32`, `alpha=64`, attention + MLP target modules, six
  fully-trainable modules in `modules_to_save` (state encoder,
  action encoder, action decoder, position embedding, vlln,
  vl_self_attention) — ~2.2 GB on disk in bf16.

This is roughly **1/7** the size of a full fine-tune checkpoint
of the same base (~15 GB).

For comparison:

| Configuration | On-disk size |
|---|---|
| Full fine-tune of GR00T N1.7 | ~15 GB |
| LoRA `r=32`, attention only, no `modules_to_save` | ~90 MB |
| **LoRA `r=32`, attention + MLP + 6 `modules_to_save`** (this adapter) | **~2.2 GB** |
| LoRA `r=16`, attention only, no `modules_to_save` | ~44 MB |

If the directory is closer to 15 GB than to 2 GB, the wrapper has
fallen back to a full save and the LoRA delta has been merged or
lost.

## 4. Forward output differs with the adapter on vs. off

```
Forward output diff (adapter on vs off): 17.93
```

This is the load-bearing check.

The script runs a single forward pass with a small synthetic input
twice — once with the adapter active, once with
`disable_adapters()` — and computes the L2 difference between the
two outputs. A correctly applied LoRA changes the model's output;
a fully merged or absent LoRA does not.

The threshold is `1e-6`. The observed value (~18) is roughly **1.7 × 10⁷**
times the threshold, so the margin is wide; what matters is that
the result is unambiguously non-zero.

> The script supports both PEFT's `PeftModel` (`disable_adapter()`
> as a context manager) and the transformers-native PEFT integration
> (`disable_adapters()` / `enable_adapters()` as instance methods).
> The two APIs are not interchangeable across versions; the script
> picks whichever is available on the loaded model.

## Expected output

```
[1/4] Trainable: 528,793,728 / 3,721,568,512 (14.21%) -- PASS
[2/4] First LoRA-A norm: 3.4491                       -- PASS
[3/4] adapter_only size: 2210 MB                      -- PASS
[4/4] Forward output diff (adapter on vs off): 17.93  -- PASS

All 4 checks passed. LoRA adapter is correctly applied.
```

If any of the four does not pass, see
[`debugging_journey.md`](./debugging_journey.md) — each blocker
described there manifests as a specific check failing.
