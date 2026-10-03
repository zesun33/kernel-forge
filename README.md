# kernel-forge

> Flagship Developer CLI & Agent Runtime for GPU Kernel Engineering (CUDA & Triton).

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![CI](https://github.com/zesun33/kernel-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/zesun33/kernel-forge/actions/workflows/ci.yml)
[![Standard: Strict](https://img.shields.io/badge/engineering%20standard-strict-blueviolet)](#engineering-standard)
[![Platforms: Linux](https://img.shields.io/badge/platforms-linux-lightgrey)](#verified-on)
[![Reference CUDA: 12.5](https://img.shields.io/badge/Reference%20CUDA-12.5-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU selection: CUDA visible](https://img.shields.io/badge/GPU%20selection-CUDA%20visible-orange)](#quick-start)
[![Maintained by zesun33](https://img.shields.io/badge/maintained%20by-zesun33-0a0a0a)](https://github.com/zesun33)

Part of the **AI Agent Tooling for Hardware & ML Systems** portfolio by [Md Zesun Ahmed Mia](https://github.com/zesun33).

---

## Quick start

Python 3.9+ is required. Benchmarking additionally requires an NVIDIA driver, CUDA-capable GPU, and `nvcc` in `PATH`. Hardware discovery uses the CUDA driver API; `nvidia-smi` is optional host metadata. Install from the checkout or use `./bin/forge` directly:

```bash
python3 -m pip install .
forge doctor --json
forge init matmul --out my-kernel
forge profile my-kernel/matmul_kernel.cu --json
```

The default device is CUDA-visible ordinal **0**. `--device N` overrides the local `FORGE_DEVICE` environment variable. Invalid selections fail instead of silently choosing another device.

```bash
# Single-GPU laptop: visible device 0 is the default.
./bin/forge doctor --json

# Local cluster example: expose host GPU 4, which CUDA renumbers to visible 0.
CUDA_VISIBLE_DEVICES=4 ./bin/forge bench src/kernel_forge/templates/matmul_tiled/kernel.cu --json

# Expose host GPUs 4 and 5, then choose the second visible GPU.
CUDA_VISIBLE_DEVICES=4,5 ./bin/forge doctor --device 1 --json

# Persist a local preference in your shell, not in this repository.
export FORGE_DEVICE=0
```

`device.index` is the ordinal passed to `cudaSetDevice`. `device.physical_index` is optional host information from `nvidia-smi`; it is not an execution argument. Numeric, UUID-based, and reordered `CUDA_VISIBLE_DEVICES` masks are resolved by CUDA itself. With no visible GPU, `doctor` reports `default_device_id: null`; benchmarking fails with a diagnostic.

Latency and throughput are measured by the CUDA templates. Roofline classifications use modeled memory traffic and theoretical reference specifications, not hardware-counter measurements of DRAM saturation. Unknown GPU models currently use fallback estimates (20 TFLOP/s and 500 GB/s), so interpret their ceilings as illustrative. The reference setup used an RTX A5000 with CUDA 12.5; other GPUs are selected by their discovered compute capability.

---

## 📚 ML Systems Jargon & Mental Model Dictionary

| Term | What It Means in Plain English | Why It Matters |
| :--- | :--- | :--- |
| **Host vs. Device** | CPU is the Host (Manager); GPU is the Device (Massive parallel factory). | PCIe bus transfers are slow; keep data in GPU VRAM as long as possible. |
| **Kernel** | A function executed in parallel by thousands of GPU threads simultaneously. | The core unit of accelerator software. |
| **Warp** | A hardware squad of 32 threads executing in strict lockstep (SIMT). | If threads in a warp take divergent branches (`if/else`), performance halves. |
| **Shared Memory (SRAM)** | On-chip scratchpad memory shared by threads in a block (~100 KB). | 10× faster than global DRAM; key to matrix multiplication tiling. |
| **Coalescing** | Threads in a warp accessing contiguous memory addresses in one trip. | Prevents memory bus stalls and maximizes effective bandwidth. |
| **FLOP** | Floating Point Operation (e.g. `a * b + c` is 2 FLOPs). | Measures total arithmetic work. |
| **Bandwidth (B)** | Rate of data transfer from VRAM to compute cores (GB/s). | 768 GB/s on RTX A5000. |
| **Arithmetic Intensity (I)** | Math operations per byte fetched: `I = FLOPs / Byte`. | Determines whether a kernel is memory-bound or compute-bound. |
| **Knee Point (I_knee)** | Ridge point: `I_knee = P_peak / B_peak`. | On RTX A5000: 36.16 FLOPs/Byte. Below this, adding ALUs provides zero speedup! |

---

## 🤖 Agent-First Interface (`--json`)

Each subcommand supports `--json` to produce structured JSON for AI IDEs (**Cursor**, **Windsurf**, **GitHub Copilot / OpenAI Codex**, **Claude Code**, **Google Antigravity**, **OpenCode**, **Cline**):

```json
{
  "status": "success",
  "device": {
    "index": 0,
    "physical_index": 4,
    "name": "NVIDIA RTX A5000",
    "compute_capability": "sm_86",
    "peak_fp32_tflops": 27.77,
    "peak_bandwidth_gbps": 768.0,
    "knee_point_flops_per_byte": 36.16
  },
  "operator": "vector_add",
  "latencies_ms": [0.0236, 0.0236, 0.0225],
  "p50_ms": 0.0236,
  "roofline": {
    "arithmetic_intensity": 0.0833,
    "achieved_bandwidth_gbps": 533.17,
    "regime": "memory_bound",
    "recommendations": [
      "Cache input matrices into fast On-Chip Shared Memory (SRAM tiling)",
      "Ensure global memory loads are 128-bit coalesced (float4 / int4)"
    ]
  }
}
```

---

## 🛡️ Engineering Standard: Strict 6-Gate Verification

```bash
# Full verification on a chosen host GPU (visible device 0)
CUDA_VISIBLE_DEVICES=4 ./scripts/verify.sh

# Fast / CI verification (headless environments without physical GPUs)
./scripts/verify.sh --quick
```

- **Gate 1 (Spec Lock)**: Verify `pyproject.toml`, `LICENSE`, `README.md`.
- **Gate 2 (Static Quality)**: Bytecode syntax verification across all modules.
- **Gate 3 (Unit Tests)**: Portable regression tests for device selection, CLI parsing, and Roofline math.
- **Gate 4 (Hardware Integration)**: Live compilation and execution of baseline kernels on the selected CUDA-visible GPU; `--quick` explicitly skips execution.
- **Gate 5 (Packaging & CLI)**: Executable permissions and `--help` snapshot tests.
- **Gate 6 (Agent Contract)**: Schema validation of JSON outputs.

---

## License

Apache-2.0. Copyright 2026 Md Zesun Ahmed Mia.
