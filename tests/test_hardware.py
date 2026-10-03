"""Portable discovery and selection regression tests (no GPU required)."""
import os
import unittest
from unittest.mock import patch
from kernel_forge.hardware import query_gpus, get_target_device, configured_device_id


class TestHardware(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"FORGE_DEVICE": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def visible(self, index=0, name="NVIDIA RTX A5000", bus="0000:87:00.0"):
        return {"index": index, "name": name, "compute_capability": "sm_86",
                "memory_mib": 23028, "clock_mhz": 1695, "pci_bus_id": bus}

    def test_visible_ordinal_is_independent_of_host_metadata(self):
        with patch("kernel_forge.hardware._query_cuda_devices", return_value=[self.visible()]), \
             patch("kernel_forge.hardware.shutil.which", return_value=None):
            devices = query_gpus()
            self.assertEqual(devices[0].index, 0)
            self.assertIsNone(devices[0].physical_index)
            self.assertEqual(get_target_device().compute_capability, "sm_86")

    def test_explicit_invalid_selection_does_not_fall_back(self):
        with patch("kernel_forge.hardware._query_cuda_devices", return_value=[self.visible()]), \
             patch("kernel_forge.hardware.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                get_target_device(4)

    def test_local_default_and_explicit_override(self):
        with patch.dict(os.environ, {"FORGE_DEVICE": "1"}):
            self.assertEqual(configured_device_id(), 1)
            self.assertEqual(configured_device_id(0), 0)

    def test_missing_driver_is_reported_as_no_devices(self):
        with patch("kernel_forge.hardware._query_cuda_devices", side_effect=OSError("missing driver")):
            self.assertEqual(query_gpus(), [])
            self.assertIsNone(get_target_device())

    def test_negative_selection_rejected(self):
        with self.assertRaises(ValueError):
            get_target_device(-1)
