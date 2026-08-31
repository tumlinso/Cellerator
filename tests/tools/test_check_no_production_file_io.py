#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_no_production_file_io", ROOT / "tools" / "check_no_production_file_io.py")
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class FileApiPatternTest(unittest.TestCase):
    def test_rejects_unqualified_and_global_calls(self):
        rejected = (
            "open(path, flags);",
            "value = read(fd, data, size);",
            "write(fd, data, size);",
            "auto* stream = fopen(path, mode);",
            "::read(fd, data, size);",
            "return ::write(fd, data, size);",
        )
        for line in rejected:
            with self.subTest(line=line):
                self.assertTrue(CHECKER.contains_c_posix_file_api(line))

    def test_accepts_member_calls(self):
        accepted = (
            "source.read(request, output);",
            "ptr->write(value);",
            "source . read(request, output);",
            "ptr -> write(value);",
            "(*source).read(request, output);",
            "stream.open(path);",
        )
        for line in accepted:
            with self.subTest(line=line):
                self.assertFalse(CHECKER.contains_c_posix_file_api(line))

    def test_member_call_does_not_hide_later_unqualified_call(self):
        line = "source.read(request); read(fd, output, size);"
        self.assertTrue(CHECKER.contains_c_posix_file_api(line))


if __name__ == "__main__":
    unittest.main()
