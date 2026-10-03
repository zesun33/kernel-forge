"""Hardware discovery, GPU profiling, and theoretical limit calculation."""

import ctypes
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

    physical_index: Optional[int] = None
    pci_bus_id: str = ""

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


def _query_cuda_devices() -> List[Dict]:
    """Use CUDA's driver API so indices match CUDA_VISIBLE_DEVICES ordering."""
    cuda = ctypes.CDLL("libcuda.so.1")

    def check(code):
        if code != 0:
            raise RuntimeError(f"CUDA discovery failed with driver status {code}")

    check(cuda.cuInit(0))
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)))
    devices = []
    for index in range(count.value):
        device = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(device), index))
        name = ctypes.create_string_buffer(256)
        bus = ctypes.create_string_buffer(32)
        major, minor, clock = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        memory = ctypes.c_size_t()
        check(cuda.cuDeviceGetName(name, len(name), device))
        check(cuda.cuDeviceGetPCIBusId(bus, len(bus), device))
        check(cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device))
        check(cuda.cuDeviceTotalMem_v2(ctypes.byref(memory), device))
        check(cuda.cuDeviceGetAttribute(ctypes.byref(clock), 13, device))  # CLOCK_RATE, kHz
        devices.append({"index": index, "name": name.value.decode(),
                        "compute_capability": f"sm_{major.value}{minor.value}",
                        "memory_mib": memory.value // (1024 * 1024),
                        "clock_mhz": clock.value // 1000, "pci_bus_id": bus.value.decode()})
    return devices


def query_gpus() -> List[GpuDevice]:
    """Discover CUDA-visible devices; index is the ordinal used by cudaSetDevice."""
    try:
        visible = _query_cuda_devices()
    except (OSError, RuntimeError):
        return []

    # Optional host metadata. Discovery and selection never depend on nvidia-smi.
    host = {}
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run([smi, "--query-gpu=index,pci.bus_id,power.limit",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 3:
                    try:
                        host[fields[1].lower().lstrip("0")] = (int(fields[0]), float(fields[2]))
                    except ValueError:
                        continue

    devices = []
    for info in visible:
        spec = next((value for key, value in GPU_DATABASE.items()
                     if key.lower() in info["name"].lower()), None)
        # Existing theoretical reference estimates; these are not measured peaks.
        tflops = spec["peak_fp32_tflops"] if spec else 20.0
        bw = spec["peak_bandwidth_gbps"] if spec else 500.0
        physical_index, power = host.get(info["pci_bus_id"].lower().lstrip("0"), (None, 0.0))
        devices.append(GpuDevice(
            index=info["index"], name=info["name"],
            compute_capability=info["compute_capability"],
            total_memory_mib=info["memory_mib"], max_sm_clock_mhz=info["clock_mhz"],
            power_limit_watts=power, peak_fp32_tflops=tflops, peak_bandwidth_gbps=bw,
            knee_point_flops_per_byte=round(tflops * 1000.0 / bw, 2),
            physical_index=physical_index, pci_bus_id=info["pci_bus_id"]))
    return devices


def configured_device_id(device_id: Optional[int] = None) -> int:
    """Explicit selection wins over local FORGE_DEVICE, then visible ordinal zero."""
    selected = device_id if device_id is not None else int(os.environ.get("FORGE_DEVICE", "0"))
    if selected < 0:
        raise ValueError("GPU device ID must be a nonnegative CUDA-visible ordinal")
    return selected


def get_target_device(device_id: Optional[int] = None) -> Optional[GpuDevice]:
    selected = configured_device_id(device_id)
    devices = query_gpus()
    if not devices:
        return None
    for device in devices:
        if device.index == selected:
            return device
    raise ValueError(f"CUDA-visible GPU {selected} is unavailable; visible IDs: "
                     + ", ".join(str(device.index) for device in devices))
