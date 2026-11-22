#!/usr/bin/env python3
"""Fixtures for the lexical native-style checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))
from check_native_style import audit, is_cellerator_public_include  # noqa: E402


class NativeStyleFixtures(unittest.TestCase):
    def audit_files(self, files: dict[str, str]) -> tuple[list[str], int, int]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, contents in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            return audit(root, ["src"])

    def test_scalar_intrinsic_and_cellerator_public_includes_are_allowed(self) -> None:
        errors, source_count, scanned_count = self.audit_files({
            "src/main.cpp": """
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <immintrin.h>
#include <Cellerator/Cell.h>
#include "Cellerator/DeviceView.cuh"
std::size_t count;
std::uint32_t tag;
""",
        })

        self.assertEqual(errors, [])
        self.assertEqual(source_count, 1)
        self.assertEqual(scanned_count, 1)

    def test_direct_library_headers_names_and_allocation_remain_rejected(self) -> None:
        errors, _, _ = self.audit_files({
            "src/main.cpp": """
#include <vector>
#include <stdlib.h>
#include <cstdlib>
std::vector<int> values;
void allocate() { int* p = new int; free(p); delete p; }
""",
        })

        self.assertTrue(any("system include <vector>" in error for error in errors))
        self.assertTrue(any("system include <stdlib.h>" in error for error in errors))
        self.assertTrue(any("system include <cstdlib>" in error for error in errors))
        self.assertTrue(any("std::vector" in error for error in errors))
        self.assertTrue(any("dynamic allocation/deallocation" in error for error in errors))

    def test_globally_qualified_std_namespace_alias_and_import_are_rejected(self) -> None:
        errors, _, _ = self.audit_files({
            "src/main.cpp": """
namespace s = ::std;
auto x = s::vector<int>{};
using namespace ::std;
vector<int> values;
""",
        })

        self.assertTrue(any("aliasing the std namespace is prohibited" in error for error in errors))
        self.assertTrue(any("importing the std namespace is prohibited" in error for error in errors))

    def test_character_literals_and_digit_separators_do_not_hide_later_violations(self) -> None:
        errors, _, _ = self.audit_files({
            "src/main.cpp": "void f() { char8_t a = u8'a'; wchar_t b = L'b'; char16_t c = u'd'; "
            "char32_t d = U'e'; int n = 1'000; std::vector<int> values; int* p = new int; }\n",
        })

        self.assertTrue(any("std::vector" in error for error in errors))
        self.assertTrue(any("dynamic allocation/deallocation" in error for error in errors))

    def test_local_wrapper_and_helper_are_scanned_around_cellerator_include(self) -> None:
        errors, _, scanned_count = self.audit_files({
            "src/main.cpp": '#include "wrapper.hpp"\n',
            "src/wrapper.hpp": """
#include <Cellerator/Cell.h>
#include "Celleratorish/Cell.h"
#include "helper.hpp"
std::vector<int> wrapped;
""",
            "src/Celleratorish/Cell.h": "std::vector<int> local_wrapper;\n",
            "src/helper.hpp": "void* allocate() { return malloc(8); }\n",
        })

        self.assertTrue(any(
            "src/wrapper.hpp" in error and "std::vector" in error for error in errors
        ))
        self.assertTrue(any(
            "src/Celleratorish/Cell.h" in error and "std::vector" in error for error in errors
        ))
        self.assertTrue(any(
            "src/helper.hpp" in error and "dynamic allocation" in error for error in errors
        ))
        self.assertFalse(any("Cellerator/Cell.h" in error for error in errors))
        self.assertEqual(scanned_count, 4)

    def test_cellerator_named_path_inside_ccell_root_is_still_audited(self) -> None:
        errors, source_count, scanned_count = self.audit_files({
            "src/main.cpp": '#include <Cellerator/local.hh>\n#include "Cellerator/local_quote.hh"\n',
            "Cellerator/local.hh": "#include <vector>\nstd::vector<int> local;\n",
            "Cellerator/local_quote.hh": '#include "local_helper.hh"\n',
            "Cellerator/local_helper.hh": "std::vector<int> local_helper;\n",
        })

        self.assertEqual(source_count, 1)
        self.assertEqual(scanned_count, 4)
        self.assertTrue(any(
            "Cellerator/local.hh" in error and "<vector>" in error for error in errors
        ))
        self.assertTrue(any(
            "Cellerator/local_helper.hh" in error and "std::vector" in error for error in errors
        ))

    def test_cellerator_path_in_public_include_root_is_still_audited(self) -> None:
        errors, source_count, scanned_count = self.audit_files({
            "src/main.cpp": '#include <Cellerator/local.hh>\n',
            "include/Cellerator/local.hh": "#include <vector>\nstd::vector<int> local;\n",
        })

        self.assertEqual(source_count, 1)
        self.assertEqual(scanned_count, 2)
        self.assertTrue(any(
            "include/Cellerator/local.hh" in error and "<vector>" in error for error in errors
        ))

    def test_macro_and_malformed_include_directives_are_rejected(self) -> None:
        errors, _, _ = self.audit_files({
            "src/macro.cpp": """
#include HEADER
""",
            "src/malformed.cpp": """
#include <stddef.h"
""",
            "src/missing.cpp": """
#include "missing.hpp"
""",
        })

        self.assertTrue(any(
            "src/macro.cpp" in error and "include path must be explicit" in error for error in errors
        ))
        self.assertTrue(any(
            "src/malformed.cpp" in error and "include path must be explicit" in error for error in errors
        ))
        self.assertTrue(any(
            'quoted include "missing.hpp" is unresolved' in error for error in errors
        ))

    def test_comments_and_ordinary_or_raw_string_literals_are_ignored(self) -> None:
        errors, _, _ = self.audit_files({
            "src/main.cpp": r'''#include <stddef.h> // #include <vector>
// #include <cstdlib>
/* #include <stdlib.h> */
const char* ordinary = "#include <memory>";
const char* raw = R"tag(
#include <Cellerator/../authored.hpp>
#include <vector>
)tag";
''',
        })

        self.assertEqual(errors, [])

    def test_cellerator_prefix_and_traversal_do_not_exempt_root_authored_file(self) -> None:
        self.assertTrue(is_cellerator_public_include("Cellerator/Cell.h"))
        for spelling in (
            "Celleratorish/Cell.h",
            "./Cellerator/Cell.h",
            "../Cellerator/Cell.h",
            "Cellerator/../authored.hpp",
            "Cellerator/./Cell.h",
            "Cellerator//Cell.h",
            "Cellerator/",
            "Cellerator\\Cell.h",
        ):
            with self.subTest(spelling=spelling):
                self.assertFalse(is_cellerator_public_include(spelling))

        errors, source_count, scanned_count = self.audit_files({
            "src/main.cpp": '#include "Cellerator/../authored.hpp"\n#include <Celleratorish/Cell.h>\n',
            "authored.hpp": "#include <vector>\n",
        })

        self.assertEqual(source_count, 1)
        self.assertEqual(scanned_count, 2)
        self.assertTrue(any(
            "authored.hpp:1" in error and "<vector>" in error for error in errors
        ))
        self.assertTrue(any("system include <Celleratorish/Cell.h>" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
