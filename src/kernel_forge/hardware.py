"""Hardware discovery, GPU profiling, and theoretical limit calculation."""

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# Reference database of GPU specifications (Ampere, Hopper, Ada, Volta)
GPU_DATABASE = {
    "RTX A5000": {
        "architecture": "Ampere",
        "compute_capability": "sm_86",
        "sms": 64,
        "cuda_cores_per_sm": 128,
        "total_cuda_cores": 8192,
        "boost_clock_mhz": 1695,
        "peak_fp32_tflops": 27.77,
        "memory_bus_width_bits": 384,
        "memory_clock_gbps": 16.0,
        "peak_bandwidth_gbps": 768.0,
        "knee_point_flops_per_byte": 36.16,
    },
    "RTX 3090": {
        "architecture": "Ampere",
        "compute_capability": "sm_86",
        "sms": 82,
        "cuda_cores_per_sm": 128,
        "total_cuda_cores": 10496,
        "boost_clock_mhz": 1695,
        "peak_fp32_tflops": 35.58,
        "memory_bus_width_bits": 384,
        "memory_clock_gbps": 19.5,
        "peak_bandwidth_gbps": 936.2,
        "knee_point_flops_per_byte": 38.01,
    },
    "RTX 4090": {
        "architecture": "Ada Lovelace",
        "compute_capability": "sm_89",
        "sms": 128,
        "cuda_cores_per_sm": 128,
        "total_cuda_cores": 16384,
        "boost_clock_mhz": 2520,
        "peak_fp32_tflops": 82.58,
        "memory_bus_width_bits": 384,
        "memory_clock_gbps": 21.0,
        "peak_bandwidth_gbps": 1008.0,
        "knee_point_flops_per_byte": 81.92,
    },
    "A100": {
        "architecture": "Ampere",
        "compute_capability": "sm_80",
        "sms": 108,
        "cuda_cores_per_sm": 64,
        "total_cuda_cores": 6912,
        "boost_clock_mhz": 1410,
        "peak_fp32_tflops": 19.49,
        "memory_bus_width_bits": 5120,
        "memory_clock_gbps": 3.2,
        "peak_bandwidth_gbps": 2039.0,
        "knee_point_flops_per_byte": 9.56,
    },
    "H100": {
        "architecture": "Hopper",
        "compute_capability": "sm_90",
        "sms": 132,
        "cuda_cores_per_sm": 128,
        "total_cuda_cores": 16896,
        "boost_clock_mhz": 1780,
        "peak_fp32_tflops": 60.15,
        "memory_bus_width_bits": 5120,
        "memory_clock_gbps": 5.2,
        "peak_bandwidth_gbps": 3350.0,
        "knee_point_flops_per_byte": 17.95,
    },
}


@dataclass
class GpuDevice:
    index: int
    name: str
    compute_capability: str
    total_memory_mib: int
    max_sm_clock_mhz: int
    power_limit_watts: float
    peak_fp32_tflops: float
    peak_bandwidth_gbps: float
    knee_point_flops_per_byte: float

    def to_dict(self) -> Dict:
        return asdict(self)


def get_nvcc_info() -> Dict[str, Optional[str]]:
    """Detect nvcc compiler location and version."""
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return {"installed": False, "path": None, "version": None}

    try:
        res = subprocess.run(
            [nvcc_path, "--version"], capture_output=True, text=True, check=True
        )
        version_line = [
            line for line in res.stdout.splitlines() if "release" in line
        ]
        version = version_line[0].strip() if version_line else "Unknown"
        return {"installed": True, "path": nvcc_path, "version": version}
    except Exception as e:
        return {"installed": True, "path": nvcc_path, "version": str(e)}


def query_gpus() -> List[GpuDevice]:
    """Query nvidia-smi for all installed GPUs on host."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []

    try:
        cmd = [
            smi,
            "--query-gpu=index,name,compute_cap,memory.total,clocks.max.sm,power.limit",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        devices = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            name = parts[1]
            cc_str = parts[2]
            arch_sm = f"sm_{cc_str.replace('.', '')}"
            mem_mib = int(float(parts[3]))
            clock_mhz = int(float(parts[4]))
            power_w = float(parts[5])

            # Lookup theoretical specs from database or compute fallback
            matched_spec = None
            for key, spec in GPU_DATABASE.items():
                if key.lower() in name.lower():
                    matched_spec = spec
                    break

            if matched_spec:
                tflops = matched_spec["peak_fp32_tflops"]
                bw = matched_spec["peak_bandwidth_gbps"]
                knee = matched_spec["knee_point_flops_per_byte"]
            else:
                # Conservative fallback estimation
                tflops = 20.0
                bw = 500.0
                knee = round(tflops * 1000.0 / bw, 2)

            dev = GpuDevice(
                index=idx,
                name=name,
                compute_capability=arch_sm,
                total_memory_mib=mem_mib,
                max_sm_clock_mhz=clock_mhz,
                power_limit_watts=power_w,
                peak_fp32_tflops=tflops,
                peak_bandwidth_gbps=bw,
                knee_point_flops_per_byte=knee,
            )
            devices.append(dev)
        return devices
    except Exception:
        return []


def get_target_device(device_id: int = 4) -> Optional[GpuDevice]:
    """Get specific GPU device. Defaults to GPU 4 (from approved user cluster: 4, 5, 6, 7)."""
    devices = query_gpus()
    for dev in devices:
        if dev.index == device_id:
            return dev
    if devices:
        return devices[0]
    return None
