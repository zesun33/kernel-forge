"""Unit tests for Roofline model mathematical analysis."""

import unittest
from kernel_forge.hardware import GpuDevice
from kernel_forge.roofline import RooflineAnalysis, analyze_roofline


class TestRoofline(unittest.TestCase):
    def setUp(self):
        self.mock_gpu = GpuDevice(
            index=4,
            name="NVIDIA RTX A5000",
            compute_capability="sm_86",
            total_memory_mib=23028,
            max_sm_clock_mhz=2100,
            power_limit_watts=230.0,
            peak_fp32_tflops=27.77,
            peak_bandwidth_gbps=768.0,
            knee_point_flops_per_byte=36.16,
        )

    def test_vector_add_arithmetic_intensity(self):
        # N = 1000000: 1M FLOPs, 12MB DRAM -> I = 1/12 = 0.0833
        analysis = analyze_roofline(
            operator="vector_add",
            problem_size={"N": 1000000},
            latency_ms=0.02,
            device=self.mock_gpu,
        )
        self.assertAlmostEqual(analysis.arithmetic_intensity, 0.0833, places=3)
        self.assertEqual(analysis.regime, "memory_bound")
        self.assertIn("DRAM Memory Bandwidth", analysis.bottleneck)

    def test_matmul_naive_vs_tiled(self):
        # M=1024, N=1024, K=1024
        # Naive: 2G FLOPs / 8.59GB DRAM -> I ~ 0.25
        naive = analyze_roofline(
            operator="matmul",
            problem_size={"M": 1024, "N": 1024, "K": 1024},
            latency_ms=1.2,
            device=self.mock_gpu,
            tiled=False,
        )
        self.assertAlmostEqual(naive.arithmetic_intensity, 0.25, places=2)
        self.assertEqual(naive.regime, "memory_bound")

        # Tiled with T=16: I ~ 4.0 FLOPs/Byte (16x increase)
        tiled = analyze_roofline(
            operator="matmul",
            problem_size={"M": 1024, "N": 1024, "K": 1024},
            latency_ms=0.9,
            device=self.mock_gpu,
            tiled=True,
            tile_size=16,
        )
        self.assertGreater(tiled.arithmetic_intensity, 3.5)
        self.assertGreater(tiled.achieved_tflops, naive.achieved_tflops)

    def test_compute_bound_classification(self):
        # Generic operator with very high arithmetic intensity
        analysis = analyze_roofline(
            operator="generic",
            problem_size={"flops": 1000000000, "bytes": 10000000},  # I = 100
            latency_ms=0.1,
            device=self.mock_gpu,
        )
        self.assertGreater(analysis.arithmetic_intensity, self.mock_gpu.knee_point_flops_per_byte)
        self.assertEqual(analysis.regime, "compute_bound")


if __name__ == "__main__":
    unittest.main()
