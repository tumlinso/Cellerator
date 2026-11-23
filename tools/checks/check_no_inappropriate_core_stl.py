#!/usr/bin/env python3
"""Reject new controlled STL ownership in Cellerator production roots."""

from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import subprocess
import sys


FAMILIES = (
    "vector",
    "map",
    "unordered_map",
    "set",
    "unordered_set",
    "priority_queue",
    "shared_ptr",
    "deque",
    "list",
)
SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".cc", ".cpp", ".cu"}
PRODUCTION_ROOTS = (
    "include/Cellerator", "src", "examples/models/developmental_time",
    "examples/models/developmental_time_trajectory", "examples/models/state_reduce",
    "examples/trajectory",
)

# Explicit core ownership exceptions and existing migration debt only. Values
# are non-increasing ceilings, not quotas. Every entry names its ownership lane.
ALLOWLIST: dict[str, dict[str, tuple[int, str]]] = {
    "examples/trajectory/trajectory_build.cuh": {"vector": (1, "CE-PTR-10 example-facing graph migration")},
    "examples/trajectory/trajectory_query.cuh": {"vector": (6, "CE-PTR-10 caller-owned query views")},
    "include/Cellerator/compute/operators/graph/forward_candidates.cuh": {"vector": (3, "CE-PTR-10 bounded edge table")},
    "include/Cellerator/compute/operators/graph/forward_prune.cuh": {"vector": (10, "CE-PTR-10 fixed-width graph relation")},
    "include/Cellerator/compute/operators/graph/incremental_insert.cuh": {"vector": (5, "CE-PTR-10 slab assignment relation")},
    "include/Cellerator/compute/operators/graph/record_table.cuh": {"vector": (12, "CE-PTR-10 trajectory record image")},
    "include/Cellerator/compute/operators/graph/slab_index.cuh": {"vector": (7, "CE-PTR-10 embryo span table")},
    "include/Cellerator/compute/operators/graph/supernode_reduce.cuh": {"vector": (27, "CE-PTR-10 topology and member images")},
    "include/Cellerator/compute/sampling.hh": {"vector": (7, "CE-PTR-07 sample image and workspace")},
    "include/Cellerator/geometry/gating.hh": {"vector": (2, "CE-PTR-04 region selection table")},
    "include/Cellerator/geometry/gating_cuda.cuh": {"vector": (4, "CE-PTR-04 compiled route image")},
    "include/Cellerator/geometry/layout_metrics.hh": {"vector": (1, "CE-PTR-04 exact metrics table")},
    "include/Cellerator/geometry/layout_selector.hh": {"vector": (1, "CE-PTR-04 region selection table")},
    "include/Cellerator/geometry/pack.hh": {"vector": (4, "CE-PTR-04 direct image compilation")},
    "include/Cellerator/geometry/planner.hh": {"vector": (11, "CE-PTR-03 static plan image")},
    "include/Cellerator/memory/cuda/shared_buffer.hh": {"shared_ptr": (2, "M05 retained handle over canonical DeviceBuffer")},
    "src/compute/dataset/sampling.cc": {"vector": (18, "CE-PTR-07 sample workspaces"), "priority_queue": (2, "CE-PTR-07 bounded selection")},
    "src/compute/dataset/sampling_materialization.cc": {"vector": (5, "CE-PTR-07 sampled CSR image")},
    "src/compute/neighbors/scoring/cuvs_sharded_knn.cu": {"vector": (1, "CE-PTR-12 move or retire legacy staging")},
    "src/layout/physical_feature_major.cc": {"vector": (1, "CE-PTR-14 projection builder workspace")},
    "src/layout/physical_transpose.cc": {"vector": (2, "CE-PTR-14 projection builder workspace")},
    "src/geometry/candidate_discovery/gene_candidate_discovery.cc": {"vector": (5, "CE-PTR-09 resident candidate pipeline")},
    "src/geometry/candidate_relation.cc": {"vector": (2, "CE-PTR-04 packed candidate relation")},
    "src/geometry/layout_metrics.cc": {"vector": (1, "CE-PTR-04 metrics workspace")},
    "src/geometry/optimizer.cc": {"vector": (2, "CE-PTR-05/06 prepared optimizer tables")},
    "src/geometry/optimizer_state.hh": {"vector": (20, "CE-PTR-05 bounded state representation")},
    "src/geometry/planner.cc": {"vector": (7, "CE-PTR-03 static plan image")},
}


def strip_comments_and_literals(text: str) -> str:
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
    return token.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def occurrences(path: pathlib.Path) -> dict[str, int]:
    text = strip_comments_and_literals(path.read_text(encoding="utf-8", errors="replace"))
    return {
        family: len(re.findall(rf"\bstd::{family}\s*<", text))
        for family in FAMILIES
    }


def audit(repo_root: pathlib.Path) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()
    for root_name in PRODUCTION_ROOTS:
        root = repo_root / root_name
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(repo_root).as_posix()
            counts = occurrences(path)
            allowed = ALLOWLIST.get(relative, {})
            for family, count in counts.items():
                if count == 0:
                    continue
                seen.add(relative)
                if family not in allowed:
                    failures.append(f"{relative}: unallowlisted std::{family} owner ({count})")
                    continue
                ceiling, rationale = allowed[family]
                if count > ceiling:
                    failures.append(
                        f"{relative}: std::{family} count {count} exceeds ceiling {ceiling} ({rationale})"
                    )
            for family in allowed:
                if counts.get(family, 0) == 0:
                    # Debt removal is accepted; stale entries are reported only in strict mode.
                    continue
    return failures


def stale_entries(repo_root: pathlib.Path) -> list[str]:
    stale: list[str] = []
    for relative, families in ALLOWLIST.items():
        path = repo_root / relative
        counts = occurrences(path) if path.is_file() else {}
        for family, (_, rationale) in families.items():
            if counts.get(family, 0) == 0:
                stale.append(f"{relative}: stale std::{family} allowlist entry ({rationale})")
    return stale



# Patch mode covers the shipped native source surface, including modules and
# examples. Retired source and explicit integration/vendor boundaries are not
# native core. The legacy full-tree audit above intentionally retains its scope.
PATCH_ROOTS = ("include/Cellerator/", "src/", "modules/", "examples/")
PATCH_EXCLUDED = (
    "include/Cellerator/interop/cellshard/", "src/velocity/",
    "src/ops/velocity/", "src/preprocess/",
)
PATCH_SUFFIXES = SOURCE_SUFFIXES | {
    ".c", ".ccm", ".cppm", ".ixx", ".mpp", ".hxx", ".cxx", ".inc", ".inl", ".ipp", ".tpp",
}
SCALAR_NAMES = set("""
    byte size_t ptrdiff_t max_align_t int8_t uint8_t int16_t uint16_t
    int32_t uint32_t int64_t uint64_t intmax_t uintmax_t uintptr_t intptr_t
""".split())
# Names used to catch unqualified usage under pre-existing using-directives.
# New directives themselves are rejected; this is not a C++ name resolver.
IMPORTED_NAMES = set("""
    tuple pair make_tuple make_pair is_integral is_same
    vector vector_bool array deque list forward_list map multimap unordered_map
    unordered_multimap set multiset unordered_set unordered_multiset stack queue
    priority_queue basic_string string wstring u8string u16string u32string
    basic_string_view string_view wstring_view u8string_view u16string_view u32string_view
    span optional any variant function move_only_function copyable_function
    function_ref reference_wrapper bind bind_front bind_back ref cref
    unique_ptr shared_ptr weak_ptr make_unique make_shared allocate_shared
    allocator allocator_traits polymorphic_allocator
    sort stable_sort partial_sort nth_element binary_search lower_bound upper_bound
    equal_range find find_if find_if_not count count_if search search_n
    for_each transform copy copy_if copy_n move move_backward fill fill_n
    generate generate_n remove remove_if remove_copy remove_copy_if unique
    unique_copy reverse rotate partition stable_partition merge inplace_merge
    set_union set_intersection set_difference set_symmetric_difference
    min_element max_element minmax_element all_of any_of none_of accumulate
    reduce transform_reduce inclusive_scan exclusive_scan transform_inclusive_scan
    transform_exclusive_scan iota ranges adjacent_find mismatch equal lexicographical_compare
    is_sorted is_sorted_until is_partitioned partition_copy partition_point
    is_heap is_heap_until make_heap push_heap pop_heap sort_heap
    min max minmax clamp swap swap_ranges iter_swap exchange
    partial_sort_copy next_permutation prev_permutation sample shuffle
    copy_backward shift_left shift_right reverse_copy rotate_copy
    distance advance next prev
""".split())
BANNED_HEADERS = set("""
    algorithm numeric ranges vector array deque list forward_list map set
    unordered_map unordered_set stack queue string string_view span optional any
    variant functional memory memory_resource iterator execution tuple utility type_traits
""".split())


RAW_LITERAL = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
NUMBER = re.compile(r"[0-9][\w.']*")
ORDINARY_LITERAL = re.compile(r"(?:u8|u|U|L)?([\"'])")


def mask_cpp(text: str) -> str:
    """Mask comments and C++ literals, retaining offsets and newlines."""
    result = list(text)
    n = len(text)
    i = 0
    while i < n:
        end = i
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            if end < 0:
                end = n
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
        elif text[i].isdigit() and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            # Skip preprocessing-number tokens so digit separators are not
            # interpreted as opening character literals.
            number = NUMBER.match(text, i)
            i = number.end()
            continue
        else:
            # Prefixes are consumed too; raw delimiters may contain any valid
            # delimiter character and the raw body can contain quotes/newlines.
            raw = RAW_LITERAL.match(text, i)
            literal = ORDINARY_LITERAL.match(text, i)
            if raw:
                marker = ")" + raw.group(1) + '"'
                end = text.find(marker, raw.end())
                end = n if end < 0 else end + len(marker)
            elif literal:
                quote = literal.group(1)
                end = literal.end()
                while end < n:
                    if text[end] == "\\":
                        end += 2
                    elif text[end] == quote:
                        end += 1
                        break
                    else:
                        end += 1
                end = min(end, n)
        if end > i:
            for j in range(i, end):
                if text[j] != "\n":
                    result[j] = " "
            i = end
        else:
            i += 1
    return "".join(result)


def git(repo: pathlib.Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def patch_source(name: str) -> bool:
    return (name.startswith(PATCH_ROOTS) and not name.startswith(PATCH_EXCLUDED)
            and not any(part in {"test", "tests", "third_party", "thirdparty", "vendor"}
                        for part in pathlib.PurePosixPath(name).parts)
            and pathlib.PurePosixPath(name).suffix in PATCH_SUFFIXES)


def include_directives(text: str):
    """Yield literal includes and their physical line spans after splicing."""
    chars = []
    line_numbers = []
    line = 1
    i = 0
    while i < len(text):
        splice = re.match(r"\\\r?\n", text[i:i + 3])
        if splice:
            i += splice.end()
            line += 1
            continue
        chars.append(text[i])
        line_numbers.append(line)
        if text[i] == "\n":
            line += 1
        i += 1
    logical = "".join(chars)
    code = mask_cpp(logical)
    for match in re.finditer(r"(?m)^[ \t]*#[ \t]*include\b", code):
        i = match.end()
        while i < len(logical):
            if logical[i] in " \t\r\v\f":
                i += 1
            elif logical.startswith("/*", i):
                end = logical.find("*/", i + 2)
                if end < 0 or "\n" in logical[i:end + 2]:
                    break
                i = end + 2
            else:
                break
        if i >= len(logical) or logical[i] not in '<"':
            continue
        close = ">" if logical[i] == "<" else '"'
        end = logical.find(close, i + 1)
        if end < 0 or "\n" in logical[i:end]:
            continue
        yield logical[i + 1:end], line_numbers[match.start()], line_numbers[end]


def patch_audit(repo: pathlib.Path, ref: str) -> list[str]:
    try:
        base = git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot resolve base {ref!r} to a commit") from exc
    print(f"CE-PTR patch base: {base}")
    # Rename detection retains old content when Git can establish the relation.
    fields = git(repo, "diff", "--name-status", "-z", "--find-renames", base, "--").split(b"\0")
    renamed = {}
    names = set()
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i].decode("utf-8", "surrogateescape")
        i += 1
        if status.startswith(("R", "C")):
            old = fields[i].decode("utf-8", "surrogateescape")
            new = fields[i + 1].decode("utf-8", "surrogateescape")
            renamed[new] = old
            names.add(fields[i + 1])
            i += 2
        else:
            names.add(fields[i])
            i += 1
    names.update(git(repo, "ls-files", "-z", "--others", "--exclude-standard").split(b"\0"))
    failures = []
    scanned = 0
    qualified = re.compile(r"\bstd\s*::\s*([A-Za-z_]\w*)")
    for raw in sorted(set(names)):
        if not raw:
            continue
        name = raw.decode("utf-8", "surrogateescape")
        path = repo / name
        if not patch_source(name) or not path.is_file():
            continue
        current = path.read_text(encoding="utf-8", errors="replace")
        oldname = renamed.get(name, name)
        try:
            old = git(repo, "show", f"{base}:{oldname}").decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            old = ""
        if current == old:
            continue
        before, after = old.splitlines(keepends=True), current.splitlines(keepends=True)
        added = set()
        for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag in {"insert", "replace"}:
                added.update(range(j1 + 1, j2 + 1))
        scanned += 1
        print(f"CE-PTR patch scanned: {name}")
        masked = mask_cpp(current)
        def report_matches(pattern: re.Pattern[str], message: str, scalar_group: int = 0) -> None:
            for match in pattern.finditer(masked):
                if scalar_group and match.group(scalar_group) in SCALAR_NAMES:
                    continue
                first = masked.count("\n", 0, match.start()) + 1
                last = masked.count("\n", 0, match.end()) + 1
                hit = next((line for line in range(first, last + 1) if line in added), None)
                if hit is not None:
                    failures.append(f"{name}:{hit}: {message}: {match.group().strip()}")
        report_matches(qualified, "prohibited standard library use", 1)
        aliases = re.findall(r"\bnamespace\s+(\w+)\s*=\s*(?:::)?\s*std\s*;", masked)
        if aliases:
            aliases_re = "|".join(re.escape(alias) for alias in aliases)
            report_matches(re.compile(r"\b(?:" + aliases_re + r")\s*::\s*([A-Za-z_]\w*)"),
                           "prohibited standard library use", 1)
        report_matches(re.compile(r"\bnamespace\s+[A-Za-z_]\w*\s*=\s*(?:::)?\s*std\s*;"), "standard namespace alias")
        report_matches(re.compile(r"\busing\s+namespace\s+(?:::)?\s*std\s*;"), "standard namespace import")
        report_matches(re.compile(r"\b(?:export\s+)?import\s+(?:std(?:\s*\.[\w.]+)?|<[^>]+>)\s*;"), "standard module import")
        for header, first, last in include_directives(current):
            hit = next((line for line in range(first, last + 1) if line in added), None)
            if hit is not None and header in BANNED_HEADERS:
                failures.append(f"{name}:{hit}: prohibited standard header <{header}>")
        # Unqualified names are dangerous only when this file imports them.
        imported = set()
        if re.search(r"\busing\s+namespace\s+(?:::)?\s*std\s*;", masked):
            imported.update(IMPORTED_NAMES)
        imported.update(name for name in re.findall(r"\busing\s+(?:::)?\s*std\s*::\s*([A-Za-z_]\w*)", masked)
                        if name not in SCALAR_NAMES)
        if imported:
            names_re = "|".join(sorted(imported, key=len, reverse=True))
            report_matches(re.compile(r"(?<![\w:])(?:" + names_re + r")\s*(?=[<(])"),
                           "imported standard library use")
    print(f"CE-PTR patch: {scanned} changed or added production files scanned; {len(failures)} violations")
    return failures

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--strict-stale", action="store_true", help="also reject allowlist entries whose debt has been removed")
    parser.add_argument("--base", help="check added lines against this Git commit")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.base:
        if args.strict_stale:
            parser.error("--strict-stale is only available for the full-tree audit")
        try:
            failures = patch_audit(repo_root, args.base)
        except (ValueError, subprocess.CalledProcessError) as exc:
            print(f"CE-PTR patch error: {exc}", file=sys.stderr)
            return 2
    else:
        failures = audit(repo_root)
    if args.strict_stale:
        failures.extend(stale_entries(repo_root))
    if failures:
        for failure in failures:
            print(f"CE-PTR STL policy violation: {failure}", file=sys.stderr)
        return 1
    if not args.base:
        print("CE-PTR STL policy: production ownership is within non-increasing allowlist ceilings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
