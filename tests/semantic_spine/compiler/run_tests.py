#!/usr/bin/env python3
"""Build and execute the bounded source-origin conformance tests (host only)."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
COMMON = [
    "src/compute/operation/relation_semantics.cc",
    "src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc",
    "src/compiler/ir/semantic/implement_relation_ir_types.cc",
    "src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc",
    "src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc",
]
BRIDGE = [
    "src/compiler/sema/relation_spine_bridge.cc",
    "src/compiler/sema/implement_numerical_tuple_semantics.cc",
    "src/compiler/frontend/parser/parse_compiler_semantic_declarations.cc",
    "src/compiler/frontend/parser/parse_biological_type_constructors_and_qualifiers.cc",
    "src/compiler/frontend/parser/parse_relation_application.cc",
    "src/compiler/frontend/parser/parse_non_relation_operation_families.cc",
]
TESTS = {
    "lowering": ("tests/semantic_spine/compiler/lowering_test.cc", COMMON),
    "source_slice": ("tests/semantic_spine/compiler/source_slice_test.cc", COMMON + BRIDGE),
    "diagnostic": ("tests/semantic_spine/compiler/diagnostic_test.cc", COMMON + BRIDGE),
    "legacy_relation_ir": (
        "tests/compiler/semantic_ir/implement_relation_apply_and_transpose_operations_test.cc", COMMON),
}
SYMBOLS = [
    "Cellerator::compiler::frontend::parser::parse_semantic_declarations_v1",
    "Cellerator::compiler::frontend::parser::parse_biological_type_v1",
    "Cellerator::compiler::frontend::parser::parse_relation_applications_v1",
    "Cellerator::compiler::frontend::parser::parse_operation_families_v1",
    "cellerator::compiler::sema::v1::valid_numerical_tuple",
    "Cellerator::compiler::sema::lower_relation_source_slice_v1",
    "Cellerator::compiler::ir::semantic::lower_relation_apply_operation_v1",
    "cellerator::compute::relation::validate",
    "cellerator::compute::relation::equivalent",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, help="Write actual commands and results as JSON")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    cuda_root = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
    if not (cuda_root / "include").is_dir():
        raise SystemExit("CUDA 12.9 headers unavailable; consult ENVIRONMENT_NOTES.md")
    build = args.build_dir.resolve() if args.build_dir else Path(tempfile.mkdtemp(prefix="ce-ss1-frontend-"))
    build.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-UNDEBUG", "-I" + str(ROOT / "include"), "-I" + str(cuda_root / "include")]
    sources = sorted(set(COMMON + BRIDGE + [test[0] for test in TESTS.values()]))
    objects = {source: build / (source.replace("/", "_") + ".o") for source in sources}
    compile_commands = [compiler + flags + ["-c", str(ROOT / source), "-o", str(objects[source])] for source in sources]
    # Independent source compilation uses all available processors, as AGENTS.md
    # requires; no device execution or benchmark resource is involved.
    jobs = os.cpu_count() or 1
    def run(command):
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(shlex.join(command) + "\n" + process.stdout + process.stderr)
        return {"command": command, "exit_code": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        compiles = list(pool.map(run, compile_commands))
    results = []
    for name, (test, dependencies) in TESTS.items():
        executable = build / name
        link = run(compiler + [str(objects[source]) for source in [test] + dependencies] + ["-o", str(executable)])
        execution = run([str(executable)])
        results.append({"name": name, "source": test, "link": link, "execution": execution})
        print(name + ": passed")
    receipt = {
        "schema_version": 1,
        "task": "CE-SS1-F04",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "compiler": run(compiler + ["--version"])["stdout"].splitlines()[0],
        "cuda_headers": str(cuda_root / "include"),
        "parallel_jobs": jobs,
        "source_sha256": {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sources},
        "symbols_exercised": SYMBOLS,
        "compile_results": compiles,
        "tests": results,
        "status": "passed",
        "scope": "Bounded embedded declarations and one assigned relation application through existing parser/Sema/IR to canonical descriptor.",
        "negative_coverage": ["altered direction with wrong endpoints", "unknown symbol", "numeric declaration mismatch", "ambiguous binding", "unconsumed source", "unsupported filter", "high identity bits", "axis/order mismatch", "invalid effects", "unknown extent", "nonempty support with empty axes", "unsupported update"],
        "accelerator_execution": "Not performed by this host conformance harness; integrated native cross-origin tests are separately required.",
        "deferred": ["full .cell executable pipeline", "full expression or execution-field lowering", "LTO and driver completion", "installed SDK completion"],
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print("receipt: " + str(args.receipt))

if __name__ == "__main__":
    main()
