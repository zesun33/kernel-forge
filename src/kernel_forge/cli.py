"""Main CLI dispatcher for kernel-forge."""

import argparse
import json
import os
import shutil
import sys
from typing import List, Optional
from kernel_forge.hardware import configured_device_id, get_nvcc_info, get_target_device, query_gpus
from kernel_forge import __version__
from kernel_forge.runner import run_benchmark


def print_doctor_human(gpus, nvcc_info, target_id=None):
    print("\n\033[1;36m=== kernel-forge Doctor: GPU Hardware & Toolchain ===\033[0m")
    if nvcc_info["installed"]:
        print(f"  \033[1;32m✓\033[0m NVCC: {nvcc_info['version']} ({nvcc_info['path']})")
    else:
        print("  \033[1;31m✗\033[0m NVCC: Not detected in PATH")

    print(f"\n  Detected {len(gpus)} CUDA-visible GPU device(s):")
    print("  " + "-" * 75)
    print(f"  {'ID':<4} {'Name':<20} {'Arch':<8} {'VRAM (MiB)':<12} {'Peak TFLOPS':<14} {'Peak BW':<10} {'Knee Point'}")
    print("  " + "-" * 75)
    for g in gpus:
        marker = " \033[1;33m[TARGET]\033[0m" if g.index == target_id else ""
        print(
            f"  {g.index:<4} {g.name:<20} {g.compute_capability:<8} {g.total_memory_mib:<12} "
            f"{g.peak_fp32_tflops:<14.2f} {g.peak_bandwidth_gbps:<10.1f} {g.knee_point_flops_per_byte:<8.2f}{marker}"
        )
    print("  " + "-" * 75)
    print(f"  Selected CUDA-visible device: {target_id}\n")


def cmd_doctor(args):
    gpus = query_gpus()
    nvcc_info = get_nvcc_info()
    requested_id = configured_device_id(args.device)
    target = get_target_device(requested_id)
    target_id = target.index if target else None

    if args.json:
        data = {
            "nvcc": nvcc_info,
            "default_device_id": target_id,
            "requested_device_id": requested_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_count": len(gpus),
            "devices": [g.to_dict() for g in gpus],
        }
        print(json.dumps(data, indent=2))
    else:
        print_doctor_human(gpus, nvcc_info, target_id=target_id)


def cmd_init(args):
    op = args.operator.lower()
    mapping = {
        "vector_add": "vector_add",
        "matmul": "matmul_tiled",
        "matmul_tiled": "matmul_tiled",
        "matmul_naive": "matmul_naive",
    }
    if op not in mapping:
        print(f"\033[1;31mError:\033[0m Unknown operator '{op}'. Available: vector_add, matmul_naive, matmul_tiled, matmul", file=sys.stderr)
        sys.exit(1)

    template_dir = os.path.join(os.path.dirname(__file__), "templates", mapping[op])
    out_dir = args.out or "."
    os.makedirs(out_dir, exist_ok=True)
    target_cu = os.path.join(out_dir, f"{op}_kernel.cu")

    src_cu = os.path.join(template_dir, "kernel.cu")
    shutil.copyfile(src_cu, target_cu)

    if args.json:
        print(json.dumps({"status": "created", "file": target_cu, "operator": op}))
    else:
        print(f"\033[1;32m✓\033[0m Initialized {op} template at: {target_cu}")


def cmd_bench(args):
    problem_params = [int(p) for p in args.params] if args.params else None
    res = run_benchmark(
        source_path=args.file,
        operator=args.op,
        device_id=args.device,
        iters=args.iters,
        problem_params=problem_params,
        arch=args.arch,
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"\n\033[1;36m=== kernel-forge Benchmark Results ===\033[0m")
        print(f"  Kernel:       {res.source_file}")
        print(f"  Device:       GPU {res.device['index']} ({res.device['name']}, {res.device['compute_capability']})")
        print(f"  Iterations:   {res.iterations}")
        print(f"  Latency p50:  \033[1;32m{res.p50_ms:.4f} ms\033[0m (p90: {res.p90_ms:.4f} ms, p99: {res.p99_ms:.4f} ms)")
        print(f"  Achieved:     {res.roofline['achieved_tflops']:.3f} TFLOPS | {res.roofline['achieved_bandwidth_gbps']:.1f} GB/s")
        print(f"  Validation:   \033[1;32mPASS\033[0m\n")


def cmd_profile(args):
    problem_params = [int(p) for p in args.params] if args.params else None
    res = run_benchmark(
        source_path=args.file,
        operator=args.op,
        device_id=args.device,
        iters=args.iters,
        problem_params=problem_params,
        arch=args.arch,
    )
    rf = res.roofline

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"\n\033[1;36m=== kernel-forge Roofline Profile ===\033[0m")
        print(f"  Operator:              {res.operator} ({res.subtype or 'standard'})")
        print(f"  Problem Size:          {res.params}")
        print(f"  Arithmetic Intensity:  \033[1;33m{rf['arithmetic_intensity']:.4f} FLOPs/Byte\033[0m")
        print(f"  Hardware Knee Point:   {rf['device_knee_point']:.2f} FLOPs/Byte")
        print(f"  Regime Classification: \033[1;35m{rf['regime'].upper()}\033[0m")
        print(f"  Achieved vs Peak:      {rf['achieved_tflops']:.3f} of {rf['device_peak_tflops']:.1f} TFLOPS ({rf['efficiency_percent']}%)")
        print(f"\n  \033[1;31mPrimary Bottleneck:\033[0m\n    {rf['bottleneck']}")
        print(f"\n  \033[1;32mOptimization Recommendations:\033[0m")
        for i, rec in enumerate(rf['recommendations'], 1):
            print(f"    {i}. {rec}")
        print()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Flagship Developer CLI & Agent Runtime for GPU Kernel Engineering (CUDA & Triton)",
    )
    parser.add_argument("-v", "--version", action="version", version=f"kernel-forge {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Probe host GPU devices, compute limits, and nvcc toolchain")
    p_doc.add_argument("--device", type=int, default=None, help="CUDA-visible device ID (default: FORGE_DEVICE or 0)")
    p_doc.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # init
    p_init = subparsers.add_parser("init", help="Scaffold verified starter kernel template")
    p_init.add_argument("operator", choices=["vector_add", "matmul_naive", "matmul_tiled", "matmul"], help="Operator type")
    p_init.add_argument("--out", "-o", default=".", help="Output directory")
    p_init.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # bench
    p_bench = subparsers.add_parser("bench", help="Compile and benchmark CUDA kernel with latency percentiles")
    p_bench.add_argument("file", help="Path to .cu kernel source file")
    p_bench.add_argument("--device", type=int, default=None, help="CUDA-visible device ID (default: FORGE_DEVICE or 0)")
    p_bench.add_argument("--iters", type=int, default=20, help="Benchmark iterations (default: 20)")
    p_bench.add_argument("--op", choices=["vector_add", "matmul"], default=None, help="Operator override")
    p_bench.add_argument("--arch", default=None, help="Target architecture (default: device capability e.g. sm_86)")
    p_bench.add_argument("--params", nargs="*", default=None, help="Problem dimension parameters")
    p_bench.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # profile
    p_prof = subparsers.add_parser("profile", help="Run Roofline arithmetic intensity & bottleneck analysis")
    p_prof.add_argument("file", help="Path to .cu kernel source file")
    p_prof.add_argument("--device", type=int, default=None, help="CUDA-visible device ID (default: FORGE_DEVICE or 0)")
    p_prof.add_argument("--iters", type=int, default=15, help="Benchmark iterations (default: 15)")
    p_prof.add_argument("--op", choices=["vector_add", "matmul"], default=None, help="Operator override")
    p_prof.add_argument("--arch", default=None, help="Target architecture (default: device capability)")
    p_prof.add_argument("--params", nargs="*", default=None, help="Problem dimension parameters")
    p_prof.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    try:
        dispatch(args, parser)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"forge: {error}\n")


def dispatch(args, parser):
    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "profile":
        cmd_profile(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
