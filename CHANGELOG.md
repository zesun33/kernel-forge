# Changelog

## 0.1.1 — 2026-10-03

- Select CUDA-visible GPU 0 by default; allow `--device` and local `FORGE_DEVICE` preferences.
- Discover devices through CUDA so UUID and reordered masks match execution ordinals.
- Reject unavailable device selections instead of silently falling back.
- Remove cluster-specific defaults from templates and verification.
- Include CUDA templates in Python wheels and source distributions.
- Reject failed kernel validation and invalid timing samples; validate complete template outputs.
- Run portable discovery tests without requiring a particular GPU model.
- Distinguish modeled Roofline traffic from measured throughput in documentation.
