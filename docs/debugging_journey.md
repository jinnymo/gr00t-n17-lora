# Debugging journey: getting LoRA back on GR00T N1.7

NVIDIA's GR00T N1.7 release does not ship LoRA support. Commit
`4e62473` on the N1.6 branch removed it, and `main` / N1.7 do not
bring it back. A naive port of the N1.5 LoRA path does not run on
N1.7 — it fails at five distinct points before anything trains
correctly. This document describes each blocker and the minimal
patch that resolves it.

The end result is the wrapper in
[`launch_finetune_grootn17_lora.py`](../launch_finetune_grootn17_lora.py).
It does not modify the upstream Isaac-GR00T source tree; instead it
monkey-patches three hooks at import time.

## Background

GR00T N1.7 (3.7 B parameters) uses the standard Hugging Face
`Trainer` and a custom `Gr00tN1d7Pipeline` to assemble the model.
A correct LoRA wrapping must:

1. apply the PEFT wrap to the model that is actually instantiated
   at training time,
2. let the LoRA-wrapped model accept the dict-shaped input that
   `Gr00tN1d7.forward` expects,
3. survive the GR00T collator's `BatchFeature(data={"inputs": ...})`
   wrap,
4. be visible to `Gr00tTrainer.save_model` so the LoRA delta is
   actually written to disk.

Each of the points below is a place where one of those breaks.

## Blocker 1 — monkey-patching the wrong subclass

**Symptom.** A patch on `BasicPipeline._create_model` runs without
errors but produces no LoRA log, and `trainable_params` reports
~16.8 % of total — far above the 0.5 – 2 % typical of LoRA at
`r=16`. Inspecting the resulting checkpoint shows the backbone
`q_proj.weight` is bit-identical to the base model and only the
action head has changed. In other words, no LoRA was applied; what
got trained was the action-head selective fine-tune that GR00T
enables by default (`tune_projector=True`, `tune_vlln=True`).

**Cause.** `BasicPipeline` (`gr00t/model/base/model_pipeline.py`)
and `Gr00tN1d7Pipeline` (`gr00t/model/gr00t_n1d7/setup.py`) are
sibling subclasses of `ModelPipeline`. N1.7 instantiates
`Gr00tN1d7Pipeline`, which defines its own `_create_model`.
Patching the base class does not affect it.

**Fix.** Patch `Gr00tN1d7Pipeline._create_model` directly (and keep
the `BasicPipeline` patch as well, for compatibility with
non-N1.7 model setups):

```python
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

_orig_n1d7 = Gr00tN1d7Pipeline._create_model

def _create_model_with_lora_n1d7(self):
    model = _orig_n1d7(self)
    if _LORA_RUNTIME_CFG.get("rank", 0) > 0:
        model = get_lora_model(model, **_LORA_RUNTIME_CFG)
    return model

Gr00tN1d7Pipeline._create_model = _create_model_with_lora_n1d7
```

After this, `trainable_params` drops to a LoRA-shaped fraction.

## Blocker 2 — `task_type=FEATURE_EXTRACTION` forces an NLP signature

**Symptom.**

```
TypeError: Gr00tN1d7.forward() got an unexpected keyword argument
'input_ids'
```

**Cause.** PEFT's `PeftModelForFeatureExtraction.forward` is
hard-coded to a Hugging Face NLP signature
(`input_ids=`, `attention_mask=`, `inputs_embeds=`, …) and forwards
those keywords into the base model. `Gr00tN1d7.forward` takes a
single `inputs: dict` argument and does not accept `input_ids`.

**Fix.** Pass `task_type=None` when constructing `LoraConfig`. PEFT
then returns a generic `PeftModel` whose
`forward(*args, **kwargs)` simply delegates to the base model
without rewriting the call signature.

```python
LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=target_modules,
    bias="none",
    task_type=None,
)
```

## Blocker 3 — collator wraps the batch in `BatchFeature(data={"inputs": ...})`

**Symptom.** With `task_type=None` in place, training now reaches
the backbone but fails inside it:

```
File ".../qwen3_backbone.py", line 142, in forward
    vl_input = {k: vl_input[k] for k in keys_to_use}
KeyError: 'input_ids'
```

A debug print of the forward arguments shows:

```
[debug-fw] arg[0] type=BatchFeature, keys=['inputs']
```

**Cause.** GR00T's processor
(`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`)
wraps the actual batch dict one level deeper:

```python
return BatchFeature(data={"inputs": batch})
```

The intended call shape from Hugging Face `Trainer.compute_loss`
is `model(**inputs)`, which unpacks the `BatchFeature` and calls
`forward(inputs=batch_dict)` — i.e. the keyword `inputs=` matches
the `Gr00tN1d7.forward(inputs: dict)` signature.

A pattern carried over from N1.5 calls `model(inputs)` (single
positional). On N1.7 that passes the entire `BatchFeature` as the
`inputs` argument, and downstream code that asks for
`inputs["input_ids"]` resolves to `BatchFeature.data["input_ids"]`,
which does not exist (only `"inputs"` does).

**Fix.** Do not override `Gr00tTrainer.compute_loss`. Use the
upstream Hugging Face `Trainer.compute_loss`, which already calls
`model(**inputs)`. The N1.5-style override should be removed
entirely.

## Blocker 4 — forward signature compatibility (resolved by the previous two fixes)

After Blocker 2 (generic `PeftModel`) and Blocker 3 (upstream
`compute_loss`), the call chain becomes:

```
Trainer.compute_loss
  → model(**inputs)                            # PeftModel.__call__
  → PeftModel.forward(**kwargs)                # task_type=None
  → LoraModel.forward(**kwargs)
  → Gr00tN1d7.forward(inputs=batch_dict)       # keyword match
```

No additional patch is needed; this blocker exists only to flag
that the combination of the previous two fixes is what makes the
PEFT wrap transparent here.

## Blocker 5 — `save_model` drops the LoRA delta

**Symptom.** Training runs to completion. Each
`checkpoint-N/` directory contains `model-*.safetensors` totalling
~13 GB, but `state_dict` contains zero `lora_A` / `lora_B` keys.
The adapter has effectively vanished at save time.

**Cause.** Hugging Face `Trainer` calls
`unwrap_model(self.model, recursive=True)` early in `train()`
(`transformers/trainer.py` around line 2459 in 4.57.x). This
strips any wrappers — including PEFT — and replaces `self.model`
with the inner base module. By the time `Gr00tTrainer.save_model`
runs, `self.model` no longer carries the PEFT layers, and only
the base `state_dict` is persisted.

`self.model_wrapped`, however, still points at the PEFT-wrapped
module (and at the DDP wrapper, when applicable).

**Fix.** Override `Gr00tTrainer.save_model` to, in addition to the
upstream save, locate the PEFT-wrapped module and call
`save_pretrained` on it into a sibling `adapter_only/` directory:

```python
_orig_save = Gr00tTrainer.save_model

def _save_model_with_lora(self, output_dir=None, _internal_call=False):
    _orig_save(self, output_dir, _internal_call)
    if output_dir is None:
        output_dir = self.args.output_dir

    candidates = [
        getattr(self, "model_wrapped", None),
        self.model,
    ]
    peft_model = None
    for m in candidates:
        if m is None:
            continue
        underlying = getattr(m, "module", m)  # unwrap DDP
        if hasattr(underlying, "peft_config"):
            peft_model = underlying
            break

    if peft_model is None:
        return

    adapter_dir = Path(output_dir) / "adapter_only"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))

Gr00tTrainer.save_model = _save_model_with_lora
```

After this, every saved checkpoint contains an
`adapter_only/adapter_model.safetensors` alongside the full
state-dict snapshot, and the adapter is independently loadable
with `PeftModel.from_pretrained(base, adapter_dir)`.

## Verification protocol

After the five fixes are in place, the wrapper is verified by the
four checks implemented in
[`verify_inference.py`](../verify_inference.py); see
[`verification.md`](./verification.md) for what each check tests
and the expected output.

## References

- N1.5 LoRA wrapper (`gr00t/utils/peft.py` on the `n1.5-release` tag) — original pattern
- N1.7 `Gr00tN1d7Pipeline` (`gr00t/model/gr00t_n1d7/setup.py`)
- N1.7 collator (`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`)
- HF Transformers 4.57.3 `Trainer` (`unwrap_model(..., recursive=True)`)
- PEFT 0.17.1 `peft_model.py` (generic vs. NLP forward variants)
- NVIDIA blog post on SO-101 + GR00T N1.5 + LoRA (rank-16 example)
