"""Ensure CUDA execution uses the discovered ordinal and rejects invalid output."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from kernel_forge.hardware import GpuDevice
from kernel_forge.runner import run_benchmark


class TestRunner(unittest.TestCase):
    def invoke(self, payload):
        device = GpuDevice(0, 'test GPU', 'sm_86', 1000, 1000, 0, 1, 1, 1000, physical_index=4)
        with patch('kernel_forge.runner.get_target_device', return_value=device), \
             patch('kernel_forge.runner.compile_cuda_kernel'), \
             patch('kernel_forge.runner.subprocess.run', return_value=SimpleNamespace(
                 returncode=0, stdout=json.dumps(payload), stderr='')) as execute:
            result = run_benchmark('vector_add.cu', iters=2)
            # Physical GPU 4 has been remapped to CUDA-visible ordinal 0.
            self.assertEqual(execute.call_args.args[0][-2:], ['0', '2'])
            return result

    def test_execute_uses_visible_ordinal(self):
        result = self.invoke({'valid': True, 'latencies_ms': [0.1, 0.2], 'params': {'N': 16}})
        self.assertEqual(result.device['physical_index'], 4)
        self.assertTrue(result.valid)

    def test_failed_or_missing_validation_never_reports_success(self):
        for valid in [False, None]:
            with self.subTest(valid=valid), self.assertRaisesRegex(RuntimeError, 'correctness'):
                self.invoke({'valid': valid, 'latencies_ms': [0.1]})

    def test_bad_latency_never_reports_success(self):
        for latency in [0, -1, float('nan'), float('inf')]:
            with self.subTest(latency=latency), self.assertRaisesRegex(RuntimeError, 'latency'):
                self.invoke({'valid': True, 'latencies_ms': [latency]})
