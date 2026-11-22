#!/usr/bin/env python3
"""Black-box checks for the Cellerator native style guard."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parents[2] / "tools/checks/check_no_inappropriate_core_stl.py"


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()

    def write(self, path, content):
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "base", "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def check(self, *args):
        return subprocess.run([sys.executable, str(CHECKER), "--repo-root", str(self.repo), *args],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_worktree_staged_untracked_and_legacy(self):
        path = "include/Cellerator/compute/sampling.hh"  # legacy vector ceiling
        self.write(path, "std::vector<int> old;\nint keep;\n")
        base = self.commit()
        self.write(path, "std::vector<int> old;\nint keep;\nstd::vector<int> staged;\n")
        self.git("add", path)
        self.write(path, "std::vector<int> old;\nint keep;\nstd::vector<int> staged;\nstd::sort(a, b);\n")
        self.write("src/new.cc", "std::optional<int> x;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(base, result.stdout)
        self.assertIn("2 changed or added production files scanned", result.stdout)
        for value in ("staged", "std::sort", "std::optional"):
            # Reported expression rather than entire source line.
            self.assertIn("std::vector" if value == "staged" else value, result.stderr)
        self.assertNotIn("sampling.hh:1:", result.stderr)

    def test_literals_comments_scalars_and_unmodified_debt(self):
        self.write("src/core.cc", "std::vector<int> old;\nint x;\n")
        base = self.commit()
        self.write("src/core.cc", '''std::vector<int> old;
int x;
std::size_t count = 0;
const char* a = u8R"tag(
std::vector<int> not_code;
)tag";
/* std::sort(a,b);
std::shared_ptr<int> x; */
// std::optional<int> x;
const char* b = "std::string";
''')
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_imports_wrappers_and_headers(self):
        base = self.commit()
        self.write("modules/core.cppm", '''#include <vector>
using namespace std;
using std::optional;
auto x = wrapper<std::unique_ptr<int>>{};
vector<int> v;
optional<int> o;
import std;
namespace st = std;
st::shared_ptr<int> p;
''')
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for value in ("<vector>", "namespace std", "std::optional", "std::unique_ptr", "vector", "import std", "st::shared_ptr"):
            self.assertIn(value, result.stderr)

    def test_removal_cannot_hide_new_debt(self):
        self.write("src/core.cc", "std::vector<int> one;\nstd::vector<int> two;\n")
        base = self.commit()
        self.write("src/core.cc", "std::vector<int> two;\nstd::vector<float> replacement;\n")
        self.assertEqual(self.check("--base", base).returncode, 1)
        self.write("src/core.cc", "std::vector<int> two;\n")
        self.assertEqual(self.check("--base", base).returncode, 0)

    def test_renamed_deleted_and_new(self):
        self.write("src/old.cc", "std::vector<int> old;\nint stable;\n")
        self.write("src/deleted.cc", "std::vector<int> deleted;\n")
        base = self.commit()
        self.git("mv", "src/old.cc", "src/renamed.cc")
        self.git("rm", "-q", "src/deleted.cc")
        self.write("examples/native_cell/new.cc", "std::span<int> x;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        self.assertIn("examples/native_cell/new.cc", result.stderr)
        self.assertNotIn("renamed.cc", result.stderr)
        self.assertNotIn("deleted.cc", result.stderr)

    def test_scalar_exceptions_and_other_std_names(self):
        base = self.commit()
        scalars = "byte size_t ptrdiff_t max_align_t int8_t uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t intmax_t uintmax_t uintptr_t intptr_t".split()
        self.write("src/core.cc", "".join(f"std::{name} value_{name};\n" for name in scalars))
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.write("src/core.cc", "std::tuple<int> x;\nstd::pair<int, int> y;\nstd::is_integral<int> z;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for name in ("tuple", "pair", "is_integral"):
            self.assertIn("std::" + name, result.stderr)

    def test_authored_boundaries_digit_separators_and_alias(self):
        base = self.commit()
        for path in ("src/interop/native.cc", "examples/experimental/example.cc", "examples/torch/example.cc"):
            self.write(path, "auto n = 1'000; std::tuple<int> forbidden;\n")
        self.write("modules/alias.cppm", "namespace s = std;\ns::pair<int, int> p;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for name in ("src/interop/native.cc", "examples/experimental/example.cc", "examples/torch/example.cc", "standard namespace alias", "s::pair"):
            self.assertIn(name, result.stderr)

    def test_native_modules_and_included_sources(self):
        self.write("modules/existing.ccm", "int old;\n")
        base = self.commit()
        self.write("modules/existing.ccm", "int old;\nstd::tuple<int> x;\n")
        self.write("modules/new.ccm", "std::pair<int, int> p;\n")
        self.write("src/new.c", "std::optional<int> x;\n")
        for suffix in (".inc", ".inl", ".ipp", ".tpp"):
            self.write("src/new" + suffix, "std::vector<int> x;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for path in ("modules/existing.ccm", "modules/new.ccm", "src/new.c", "src/new.inc", "src/new.inl", "src/new.ipp", "src/new.tpp"):
            self.assertIn(path, result.stderr)

    def test_global_std_namespace_imports_and_existing_debt(self):
        self.write("src/existing.cc", "using namespace ::std;\nnamespace old = ::std;\nusing ::std::tuple;\n")
        base = self.commit()
        self.write("src/existing.cc", "using namespace ::std;\nnamespace old = ::std;\nusing ::std::tuple;\nvector<int> x;\nold::pair<int, int> y;\ntuple<int> z;\n")
        self.write("src/new.cc", "namespace s = ::std;\nauto x = s::vector<int>{};\nusing namespace ::std;\nvector<int> y;\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for item in ("standard namespace alias", "standard namespace import", "s::vector", "existing.cc:4:", "old::pair", "existing.cc:6:"):
            self.assertIn(item, result.stderr)

    def test_include_comments_splices_and_continuation_changes(self):
        self.write("src/changed.cc", "#include <vec\\\nnot_vector>\n")
        base = self.commit()
        self.write("src/changed.cc", "#include <vec\\\ntor>\n")
        self.write("src/comment.cc", "#include/**/<vector>\n")
        self.write("modules/spliced.ccm", "#include <vec\\\ntor>\n")
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 1)
        for path in ("src/changed.cc:2:", "src/comment.cc:1:", "modules/spliced.ccm:1:"):
            self.assertIn(path, result.stderr)

    def test_include_like_text_in_literals_and_comments_is_ignored(self):
        base = self.commit()
        self.write("src/literals.cc", 'const char* p = u8R"raw(\n#include/**/<vector>\n)raw";\n/*\n#include <vector>\n*/\n// comment \\\n#include <vector>\n')
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_ref_and_existing_cli(self):
        self.commit()
        self.assertEqual(self.check("--base", "missing-ref").returncode, 2)
        self.assertEqual(self.check().returncode, 0)
        self.assertEqual(self.check("--strict-stale").returncode, 1)


if __name__ == "__main__":
    unittest.main()
