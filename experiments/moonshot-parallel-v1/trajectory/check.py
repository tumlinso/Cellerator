"""Run CPU trajectory reference qualification without allocating a GPU."""
from pathlib import Path
import sys
import unittest
import torch


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        raise RuntimeError("trajectory qualification discovered no tests")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
