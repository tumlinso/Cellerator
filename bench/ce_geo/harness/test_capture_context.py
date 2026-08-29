#!/usr/bin/env python3

from __future__ import annotations

import unittest

from capture_context import validate_device, validate_toolchain


class CaptureContextTest(unittest.TestCase):
    def test_bounded_controller_capture(self) -> None:
        device = {
            "uuid": "GPU-test",
            "name": "test",
            "pci_bus_id": "0000:00:00.0",
            "performance_class": "nvidia-sm70",
            "driver_version": "test",
        }
        self.assertIs(validate_device(device), device)
        toolchain = {"cxx": "test", "cuda_toolkit": "test",
                     "nvcc": "test", "cmake": "test"}
        self.assertIs(validate_toolchain(toolchain), toolchain)

    def test_process_or_secret_capture_is_rejected(self) -> None:
        device = {
            "uuid": "GPU-test",
            "name": "test",
            "pci_bus_id": "0000:00:00.0",
            "performance_class": "nvidia-sm70",
            "driver_version": "test",
            "processes": ["must-not-be-committed"],
        }
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_device(device)


if __name__ == "__main__":
    unittest.main()
