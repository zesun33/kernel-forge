"""Unit tests for kernel-forge CLI parsing and commands."""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from kernel_forge.hardware import GpuDevice
from kernel_forge.cli import build_parser, cmd_doctor, cmd_init


class TestCli(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_parser_subcommands(self):
        args = self.parser.parse_args(["doctor", "--device", "4", "--json"])
        self.assertEqual(args.command, "doctor")
        self.assertEqual(args.device, 4)
        self.assertTrue(args.json)

        args = self.parser.parse_args(["bench", "kernel.cu", "--iters", "50"])
        self.assertEqual(args.command, "bench")
        self.assertEqual(args.file, "kernel.cu")
        self.assertEqual(args.iters, 50)
        self.assertIsNone(args.device)

    def test_doctor_json_output(self):
        args = self.parser.parse_args(["doctor", "--device", "4", "--json"])
        buf = io.StringIO()
        device = GpuDevice(4, "test GPU", "sm_86", 1000, 1000, 0, 1, 1, 1000)
        with redirect_stdout(buf), patch("kernel_forge.cli.query_gpus", return_value=[device]), patch("kernel_forge.cli.get_target_device", return_value=device):
            cmd_doctor(args)
        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertIn("nvcc", data)
        self.assertIn("devices", data)
        self.assertEqual(data["default_device_id"], 4)

    def test_init_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.parser.parse_args(["init", "matmul", "--out", tmpdir, "--json"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                cmd_init(args)
            output = buf.getvalue().strip()
            data = json.loads(output)
            self.assertEqual(data["status"], "created")
            self.assertTrue(os.path.exists(data["file"]))


if __name__ == "__main__":
    unittest.main()
