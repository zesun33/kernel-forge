"""Unit tests for hardware discovery and theoretical limit calculations."""

import unittest
from kernel_forge.hardware import (
    GPU_DATABASE,
    GpuDevice,
    get_nvcc_info,
    get_target_device,
    query_gpus,
)


class TestHardware(unittest.TestCase):
    def test_gpu_database_specs(self):
        a5000 = GPU_DATABASE["RTX A5000"]
        self.assertEqual(a5000["compute_capability"], "sm_86")
        self.assertEqual(a5000["peak_fp32_tflops"], 27.77)
        self.assertEqual(a5000["peak_bandwidth_gbps"], 768.0)
        self.assertAlmostEqual(a5000["knee_point_flops_per_byte"], 36.16, places=1)

    def test_nvcc_info(self):
        info = get_nvcc_info()
        self.assertIn("installed", info)
        if info["installed"]:
            self.assertIsNotNone(info["path"])
            self.assertIn("12", info["version"])

    def test_query_gpus(self):
        devices = query_gpus()
        if devices:
            self.assertGreaterEqual(len(devices), 1)
            d0 = devices[0]
            self.assertIsInstance(d0, GpuDevice)
            self.assertIn("RTX", d0.name)
            self.assertEqual(d0.compute_capability, "sm_86")

    def test_target_device_default_4(self):
        dev = get_target_device(4)
        if dev:
            self.assertEqual(dev.index, 4)
            self.assertEqual(dev.compute_capability, "sm_86")
            self.assertAlmostEqual(dev.peak_fp32_tflops, 27.77, places=1)


if __name__ == "__main__":
    unittest.main()
