"""Roofline performance model analysis and optimization recommendations."""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from kernel_forge.hardware import GpuDevice


@dataclass
class RooflineAnalysis:
    operator: str
    problem_size: Dict[str, int]
    total_flops: int
    total_dram_bytes: int
    arithmetic_intensity: float  # FLOPs/Byte
    latency_ms: float
    achieved_tflops: float
    achieved_bandwidth_gbps: float
    device_peak_tflops: float
    device_peak_bandwidth_gbps: float
    device_knee_point: float
    attainable_tflops: float
    efficiency_percent: float
    regime: str  # "memory_bound" or "compute_bound"
    bottleneck: str
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def analyze_roofline(
    operator: str,
    problem_size: Dict[str, int],
    latency_ms: float,
    device: GpuDevice,
    tiled: bool = False,
    tile_size: int = 16,
) -> RooflineAnalysis:
    """Analyze a kernel execution using the Roofline Performance Model."""
    latency_sec = max(latency_ms * 1e-3, 1e-9)

    if operator == "vector_add":
        n = problem_size.get("N", 1024 * 1024)
        total_flops = n  # 1 addition per element
        # Reads A (4N) + reads B (4N) + writes C (4N) = 12N bytes
        total_dram_bytes = 12 * n
        arithmetic_intensity = total_flops / max(total_dram_bytes, 1)

    elif operator in ("matmul", "gemm"):
        m = problem_size.get("M", 1024)
        n = problem_size.get("N", 1024)
        k = problem_size.get("K", 1024)
        total_flops = 2 * m * n * k  # Multiply-accumulate = 2 FLOPs

        if tiled:
            # Tiled GEMM reuses data in Shared Memory:
            # Ideal DRAM access reads A (M*K*4) and B (K*N*4) and writes C (M*N*4)
            # realistically scaled by tile depth passes: (M*N*K / tile_size) ...
            # Standard practical DRAM traffic with tile size T:
            # Each block of T*T loads T*K from A and K*T from B
            blocks_m = (m + tile_size - 1) // tile_size
            blocks_n = (n + tile_size - 1) // tile_size
            dram_reads_a = blocks_m * blocks_n * (k * tile_size) * 4
            dram_reads_b = blocks_m * blocks_n * (k * tile_size) * 4
            dram_writes_c = m * n * 4
            total_dram_bytes = dram_reads_a + dram_reads_b + dram_writes_c
        else:
            # Naive GEMM: each of M*N threads reads 2*K floats directly from DRAM
            total_dram_bytes = (m * n * 2 * k * 4) + (m * n * 4)

        arithmetic_intensity = total_flops / max(total_dram_bytes, 1)

    else:
        # Generic operator fallback
        total_flops = problem_size.get("flops", 1000000)
        total_dram_bytes = problem_size.get("bytes", 1000000)
        arithmetic_intensity = total_flops / max(total_dram_bytes, 1)

    achieved_tflops = round((total_flops / latency_sec) / 1e12, 3)
    achieved_bw_gbps = round((total_dram_bytes / latency_sec) / 1e9, 2)

    # Roofline ceiling calculation:
    # P_attainable = min(P_peak, I * B_peak)
    memory_ceiling_tflops = (arithmetic_intensity * device.peak_bandwidth_gbps) / 1000.0
    attainable_tflops = min(device.peak_fp32_tflops, memory_ceiling_tflops)

    efficiency_percent = (
        round((achieved_tflops / attainable_tflops) * 100.0, 1)
        if attainable_tflops > 0
        else 0.0
    )

    if arithmetic_intensity < device.knee_point_flops_per_byte:
        regime = "memory_bound"
        bottleneck = (
            f"DRAM Memory Bandwidth saturated ({achieved_bw_gbps} GB/s of {device.peak_bandwidth_gbps} GB/s peak). "
            f"Arithmetic Intensity ({arithmetic_intensity:.3f} FLOPs/Byte) is below the hardware knee point ({device.knee_point_flops_per_byte:.1f})."
        )
        recommendations = [
            "Cache input matrices into fast On-Chip Shared Memory (SRAM tiling)",
            "Ensure global memory loads are 128-bit coalesced (float4 / int4)",
            "Fuse subsequent activation (ReLU, bias, scale) to avoid round-tripping to DRAM",
        ]
    else:
        regime = "compute_bound"
        bottleneck = (
            f"Compute Core throughput bound. Arithmetic intensity ({arithmetic_intensity:.2f} FLOPs/Byte) "
            f"exceeds knee point ({device.knee_point_flops_per_byte:.1f})."
        )
        recommendations = [
            "Unroll inner loops to maximize Instruction-Level Parallelism (ILP)",
            "Register-tile output matrix elements to avoid redundant Shared Memory bank conflicts",
            "Leverage hardware Tensor Cores via WMMA (warp-level matrix multiply-accumulate) or MMA PTX",
        ]

    return RooflineAnalysis(
        operator=operator,
        problem_size=problem_size,
        total_flops=total_flops,
        total_dram_bytes=total_dram_bytes,
        arithmetic_intensity=round(arithmetic_intensity, 4),
        latency_ms=round(latency_ms, 4),
        achieved_tflops=achieved_tflops,
        achieved_bandwidth_gbps=achieved_bw_gbps,
        device_peak_tflops=device.peak_fp32_tflops,
        device_peak_bandwidth_gbps=device.peak_bandwidth_gbps,
        device_knee_point=device.knee_point_flops_per_byte,
        attainable_tflops=round(attainable_tflops, 3),
        efficiency_percent=efficiency_percent,
        regime=regime,
        bottleneck=bottleneck,
        recommendations=recommendations,
    )
