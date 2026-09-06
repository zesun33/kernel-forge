# kernel-forge

> Flagship Developer CLI & Agent Runtime for GPU Kernel Engineering (CUDA & Triton).

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![CI](https://github.com/zesun33/kernel-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/zesun33/kernel-forge/actions/workflows/ci.yml)
[![Standard: Strict](https://img.shields.io/badge/engineering%20standard-strict-blueviolet)](#engineering-standard)
[![Platforms: Linux](https://img.shields.io/badge/platforms-linux-lightgrey)](#verified-on)
[![CUDA: 12.5](https://img.shields.io/badge/cuda-12.5%20%7C%20sm__86-green)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware: 8x RTX A5000](https://img.shields.io/badge/hardware-8x%20NVIDIA%20RTX%20A5000-orange)](#hardware-support)
[![Maintained by zesun33](https://img.shields.io/badge/maintained%20by-zesun33-0a0a0a)](https://github.com/zesun33)

Part of the **AI Agent Tooling for Hardware & ML Systems** portfolio by [Md Zesun Ahmed Mia](https://github.com/zesun33).

---

## ⚡ Quick Tour: See It in Action

`kernel-forge` bridges raw CUDA C++ and Python Triton kernels with the **Roofline Performance Model**, providing both humans and AI coding agents with instant, mathematical feedback on memory vs. compute saturation.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. HARDWARE PROBE (forge doctor)                                            │
│    Discovers all 8x NVIDIA RTX A5000 GPUs on host, NVCC 12.5 toolchain,     │
│    and theoretical compute peaks (27.8 TFLOPS FP32, 768 GB/s GDDR6, Iknee). │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ targets approved cluster (GPU 4)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. MICROBENCHMARK RUNNER (forge bench)                                      │
│    Compiles with nvcc -O3 -arch=sm_86 and runs deterministic iterations     │
│    with CUDA events, warmup passes, and latency percentiles (p50, p90, p99).│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ calculates arithmetic intensity
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ROOFLINE ANALYZER (forge profile --json)                                 │
│    Calculates Arithmetic Intensity (I = FLOPs / Byte), classifies regime    │
│    (MEMORY-BOUND vs COMPUTE-BOUND), and provides actionable optimization!   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Probe GPU Cluster (`forge doctor`)

```bash
./bin/forge doctor
```

Output:
```text
=== kernel-forge Doctor: GPU Hardware & Toolchain ===
  ✓ NVCC: Cuda compilation tools, release 12.5, V12.5.82 (/usr/local/cuda/bin/nvcc)

  Detected 8 GPU device(s) on host:
  ---------------------------------------------------------------------------
  ID   Name                 Arch     VRAM (MiB)   Peak TFLOPS    Peak BW    Knee Point
  ---------------------------------------------------------------------------
  0    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  1    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  2    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  3    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  4    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16    [TARGET]
  5    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  6    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  7    NVIDIA RTX A5000     sm_86    23028        27.77          768.0      36.16   
  ---------------------------------------------------------------------------
  Default benchmark device: GPU 4 (Approved cluster: 4, 5, 6, 7)
```

### 2. Scaffold a Verified Kernel (`forge init`)

```bash
./bin/forge init matmul --out src/my_kernels
```

Creates a complete, compilable 2D Shared-Memory Tiled matrix multiplication kernel ready for benchmarking.

### 3. Roofline Profile with Actionable Bottleneck Analysis (`forge profile`)

```bash
./bin/forge profile src/kernel_forge/templates/matmul_tiled/kernel.cu --device 4
```

Output:
```text
=== kernel-forge Roofline Profile ===
  Operator:              matmul (tiled)
  Problem Size:          {'M': 1024, 'N': 1024, 'K': 1024, 'TILE': 16}
  Arithmetic Intensity:  3.9690 FLOPs/Byte
  Hardware Knee Point:   36.16 FLOPs/Byte
  Regime Classification: MEMORY_BOUND
  Achieved vs Peak:      2.198 of 27.8 TFLOPS (72.1% of attainable memory ceiling)

  Primary Bottleneck:
    DRAM Memory Bandwidth saturated (553.86 GB/s of 768.0 GB/s peak). Arithmetic Intensity (3.969 FLOPs/Byte) is below the hardware knee point (36.2).

  Optimization Recommendations:
    1. Cache input matrices into fast On-Chip Shared Memory (SRAM tiling)
    2. Ensure global memory loads are 128-bit coalesced (float4 / int4)
    3. Fuse subsequent activation (ReLU, bias, scale) to avoid round-tripping to DRAM
```

---

## 📚 ML Systems Jargon & Mental Model Dictionary

| Term | What It Means in Plain English | Why It Matters |
| :--- | :--- | :--- |
| **Host vs. Device** | CPU is the Host (Manager); GPU is the Device (Massive parallel factory). | PCIe bus transfers are slow; keep data in GPU VRAM as long as possible. |
| **Kernel** | A function executed in parallel by thousands of GPU threads simultaneously. | The core unit of accelerator software. |
| **Warp** | A hardware squad of 32 threads executing in strict lockstep (SIMT). | If threads in a warp take divergent branches (`if/else`), performance halves. |
| **Shared Memory (SRAM)** | On-chip scratchpad memory shared by threads in a block ($\sim 100\text{ KB}$). | $10\times$ faster than global DRAM; key to matrix multiplication tiling. |
| **Coalescing** | Threads in a warp accessing contiguous memory addresses in one trip. | Prevents memory bus stalls and maximizes effective bandwidth. |
| **FLOP** | Floating Point Operation (e.g. $a \times b + c$ is 2 FLOPs). | Measures total arithmetic work. |
| **Bandwidth ($B$)** | Rate of data transfer from VRAM to compute cores ($\text{GB/s}$). | 768 GB/s on RTX A5000. |
| **Arithmetic Intensity ($I$)** | Math operations per byte fetched: $I = \frac{\text{FLOPs}}{\text{Byte}}$. | Determines whether a kernel is memory-bound or compute-bound. |
| **Knee Point ($I_{\text{knee}}$)** | Ridge point: $I_{\text{knee}} = \frac{P_{\text{peak}}}{B_{\text{peak}}}$. | On RTX A5000: $36.16\text{ FLOPs/Byte}$. Below this, adding ALUs provides zero speedup! |

---

## 🤖 Agent-First Interface (`--json`)

Every command supports `--json` to produce $<150$ tokens of structured JSON for AI IDEs (**Cursor**, **Windsurf**, **GitHub Copilot / OpenAI Codex**, **Claude Code**, **Google Antigravity**, **OpenCode**, **Cline**):

```json
{
  "status": "success",
  "device": {
    "index": 4,
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
# Full verification (with live CUDA kernel execution on GPU 4)
./scripts/verify.sh

# Fast / CI verification (headless environments without physical GPUs)
./scripts/verify.sh --quick
```

- **Gate 1 (Spec Lock)**: Verify `pyproject.toml`, `LICENSE`, `README.md`.
- **Gate 2 (Static Quality)**: Bytecode syntax verification across all modules.
- **Gate 3 (Unit Tests)**: 10/10 unit tests for hardware specs, CLI parsing, and Roofline math.
- **Gate 4 (Hardware Integration)**: Live compilation and execution of baseline kernels on GPU 4.
- **Gate 5 (Packaging & CLI)**: Executable permissions and `--help` snapshot tests.
- **Gate 6 (Agent Contract)**: Schema validation of JSON outputs.

---

## License

Apache-2.0. Copyright 2026 Md Zesun Ahmed Mia.
