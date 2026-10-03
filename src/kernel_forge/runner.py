"""CUDA compilation (nvcc) and benchmark execution runner."""

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from kernel_forge.hardware import GpuDevice, get_nvcc_info, get_target_device
from kernel_forge.roofline import RooflineAnalysis, analyze_roofline


@dataclass
class BenchmarkResult:
    status: str
    source_file: str
    device: Dict[str, Any]
    operator: str
    subtype: Optional[str]
    params: Dict[str, int]
    valid: bool
    iterations: int
    latencies_ms: List[float]
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    roofline: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_operator(source_path: str) -> str:
    """Infer operator type from filename or contents."""
    fname = os.path.basename(source_path).lower()
    if "vector" in fname or "add" in fname:
        return "vector_add"
    if "matmul" in fname or "gemm" in fname:
        return "matmul"

    try:
        with open(source_path, "r", errors="ignore") as f:
            content = f.read()
            if "vector_add" in content:
                return "vector_add"
            if "gemm" in content or "matmul" in content:
                return "matmul"
    except Exception:
        pass
    return "generic"


def compile_cuda_kernel(
    source_path: str,
    output_bin: str,
    arch: str = "sm_86",
    extra_flags: Optional[List[str]] = None,
) -> None:
    """Compile CUDA source file using host nvcc."""
    nvcc_info = get_nvcc_info()
    if not nvcc_info["installed"] or not nvcc_info["path"]:
        raise RuntimeError("NVIDIA CUDA Compiler (nvcc) not found on system.")

    nvcc = nvcc_info["path"]
    cmd = [nvcc, "-O3", f"-arch={arch}", source_path, "-o", output_bin]
    if extra_flags:
        cmd.extend(extra_flags)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{res.stderr}")


def run_benchmark(
    source_path: str,
    operator: Optional[str] = None,
    device_id: Optional[int] = None,
    iters: int = 20,
    problem_params: Optional[List[int]] = None,
    arch: Optional[str] = None,
) -> BenchmarkResult:
    """Compile, execute, and profile a CUDA kernel."""
    if iters <= 0 or (problem_params and any(value <= 0 for value in problem_params)):
        raise ValueError("Iterations and problem dimensions must be positive")
    device = get_target_device(device_id)
    if not device:
        raise RuntimeError(f"GPU device {device_id} not detected.")

    target_arch = arch or device.compute_capability
    inferred_op = operator or detect_operator(source_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "kernel_bench")
        compile_cuda_kernel(source_path, bin_path, arch=target_arch)

        # Build execution args: [problem_params...] [device_id] [iters]
        exec_cmd = [bin_path]
        if problem_params:
            for p in problem_params:
                exec_cmd.append(str(p))
        else:
            # Default sizing if none provided
            if inferred_op == "vector_add":
                exec_cmd.append("1048576")
            elif inferred_op in ("matmul", "gemm"):
                exec_cmd.extend(["1024", "1024", "1024"])

        exec_cmd.extend([str(device.index), str(iters)])

        exec_res = subprocess.run(exec_cmd, capture_output=True, text=True)
        if exec_res.returncode != 0:
            raise RuntimeError(f"Kernel execution failed:\n{exec_res.stderr}")

        # Parse output JSON line
        raw_output = exec_res.stdout.strip()
        try:
            # Find the JSON line in stdout
            json_line = [
                line for line in raw_output.splitlines() if line.startswith("{")
            ][-1]
            data = json.loads(json_line)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse kernel output JSON: {e}\nRaw stdout:\n{raw_output}"
            )

        latencies: List[float] = data.get("latencies_ms", [])
        if not latencies or any(not math.isfinite(value) or value <= 0 for value in latencies):
            raise RuntimeError("Missing or invalid latency measurements.")
        if data.get("valid") is not True:
            raise RuntimeError("Kernel correctness validation did not pass.")

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50 = latencies_sorted[int(n * 0.50)]
        p90 = latencies_sorted[min(int(n * 0.90), n - 1)]
        p99 = latencies_sorted[min(int(n * 0.99), n - 1)]
        mean_ms = round(sum(latencies) / n, 4)

        op_name = inferred_op or data.get("operator", "generic")
        subtype = data.get("subtype")
        params = data.get("params", {})
        valid = data["valid"]

        is_tiled = subtype == "tiled"
        tile_sz = params.get("TILE", 16)

        roofline = analyze_roofline(
            operator=op_name,
            problem_size=params,
            latency_ms=p50,
            device=device,
            tiled=is_tiled,
            tile_size=tile_sz,
        )

        return BenchmarkResult(
            status="success",
            source_file=source_path,
            device=device.to_dict(),
            operator=op_name,
            subtype=subtype,
            params=params,
            valid=valid,
            iterations=n,
            latencies_ms=latencies,
            p50_ms=round(p50, 4),
            p90_ms=round(p90, 4),
            p99_ms=round(p99, 4),
            mean_ms=mean_ms,
            roofline=roofline.to_dict(),
        )
