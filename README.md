# gr00t-n17-lora

LoRA fine-tuning for NVIDIA GR00T N1.7 (3B params) on a single 24 GB GPU.

NVIDIA's N1.7 release does not ship LoRA support; commit `4e62473` on
the n1.6 branch removed it, and main / N1.7 do not bring it back. This
repo restores LoRA on N1.7 by monkey-patching three hooks in
Isaac-GR00T, without modifying the upstream source tree.

Status: pre-release. Full README, results, debugging journey, and a
public Hugging Face adapter are forthcoming.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
