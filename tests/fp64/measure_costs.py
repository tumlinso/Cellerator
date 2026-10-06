#!/usr/bin/env python3
"""Build the source-bound historical FP32 CSR wrapper and run the cost harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_BASELINE = Path(__file__).resolve().parent / "reference/project_f32_baseline.cu"


def extract_function(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise ValueError(f"historical source is missing signature {signature!r}")
    brace = source.find("{", start)
    if brace < 0:
        raise ValueError(f"historical source has no body for {signature!r}")
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise ValueError(f"historical source has an unterminated body for {signature!r}")


def baseline_translation_unit(source: str) -> tuple[str, dict[str, int]]:
    kernel_signature = "__global__ void csr_spmm_fwd_f32_kernel_("
    wrapper_signature = "void csr_spmm_fwd_f32(const runtime::execution_context& ctx,"
    kernel = extract_function(source, kernel_signature)
    wrapper = extract_function(source, wrapper_signature).replace(
        "void csr_spmm_fwd_f32(", "void csr_spmm_fwd_f32_baseline(", 1
    )
    lines = source.splitlines()
    kernel_line = next(i + 1 for i, line in enumerate(lines) if kernel_signature in line)
    wrapper_line = next(i + 1 for i, line in enumerate(lines) if wrapper_signature in line)
    result = r'''#include <Cellerator/compute/candidate/sparse/project.hh>
#include <cuda_runtime.h>
#include <cstdint>

namespace cellerator::compute::sparse::project {
namespace {
constexpr int kSpmmColsThreads = 128;
''' + kernel + r'''
} // namespace
''' + wrapper + r'''
} // namespace cellerator::compute::sparse::project

extern "C" void fp64_old_csr_spmm_fwd_f32(
    const cellerator::runtime::execution_context* ctx,
    const std::uint32_t* major_ptr, const std::uint32_t* minor_idx,
    const float* values, std::uint32_t rows, std::uint32_t cols,
    const float* rhs, std::int64_t rhs_ld, std::int64_t out_cols,
    float* out, std::int64_t out_ld, const std::uint32_t* value_indices,
    float input_scale, float destination_scale) {
    cellerator::compute::sparse::project::csr_spmm_fwd_f32_baseline(
        *ctx, major_ptr, minor_idx, values, rows, cols, rhs, rhs_ld,
        out_cols, out, out_ld, value_indices, input_scale, destination_scale);
}
'''
    return result, {"kernel_start_line": kernel_line, "wrapper_start_line": wrapper_line}


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), file=sys.stderr)
    return subprocess.run(command, cwd=cwd, text=True, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--baseline-source", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", shutil.which("nvcc")))
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--build-baseline-only", action="store_true")
    args = parser.parse_args()

    if not args.baseline_source.is_file():
        raise SystemExit(f"baseline source unavailable: {args.baseline_source}; no comparison was run")
    source_bytes = args.baseline_source.read_bytes()
    source = source_bytes.decode("utf-8")
    unit, source_lines = baseline_translation_unit(source)
    if args.baseline_library is None:
        raise SystemExit("--baseline-library is required")
    sidecar = args.baseline_library.with_suffix(args.baseline_library.suffix + ".json")
    if args.build_baseline_only:
        if not args.nvcc:
            raise SystemExit("nvcc unavailable; cannot build source-bound old FP32 baseline")
        include = args.repository / "include"
        helper_include = args.repository / "src/compute/candidate/sparse"
        generated_include_candidates = ([args.binary.parent / "cellerator/generated"] if args.binary else []) + [
            args.repository / "build/generated",
        ]
        generated_include = next((path for path in generated_include_candidates if path.is_dir()), None)
        if generated_include is None:
            raise SystemExit("generated CUDA include directory not found beside benchmark build or repository build")
        args.baseline_library.parent.mkdir(parents=True, exist_ok=True)
        generated = args.baseline_library.with_suffix(".cu")
        generated.write_text(unit)
        command = [
            str(args.nvcc), "-std=c++17", "-shared", "-O3", "-DNDEBUG",
            "-Xcompiler=-fPIC", "-Xptxas=-O3", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "-gencode=arch=compute_70,code=compute_70",
            "-gencode=arch=compute_70,code=sm_70", f"-I{include}",
            f"-I{helper_include}", f"-I{generated_include}",
            str(generated), "-o", str(args.baseline_library),
        ]
        built = run(command, cwd=args.repository)
        metadata = {
            "baseline_source": str(args.baseline_source),
            "baseline_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "baseline_source_provenance": "historical source snapshot; baseline archive provenance is not asserted",
            "baseline_extracted_lines": source_lines,
            "baseline_linkage": "compiled extracted historical FP32 CSR kernel/wrapper only; no old prepared-relation ABI or archive linked",
            "compile_command": command,
            "compile_stderr": built.stderr,
        }
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"baseline_library": str(args.baseline_library),
                          "source_sha256": metadata["baseline_source_sha256"],
                          "metadata": str(sidecar)}, sort_keys=True))
        return 0
    if args.binary is None or not args.binary.is_file():
        raise SystemExit("normal measurement requires an existing --binary")
    if args.receipt is None:
        raise SystemExit("normal measurement requires --receipt")
    if not args.baseline_library.is_file() or not sidecar.is_file():
        raise SystemExit("prebuilt baseline library and its metadata sidecar are required")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("baseline_source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise SystemExit("prebuilt baseline metadata does not match the current historical source snapshot")
    executed = run([str(args.binary.resolve()), "--baseline-library", str(args.baseline_library.resolve())], cwd=args.repository)
    records = [json.loads(line) for line in executed.stdout.splitlines() if line.strip()]

    baseline = next((record for record in records if record.get("kind") == "direct_f32_baseline"), None)
    if baseline is None:
        raise SystemExit("benchmark did not emit the direct FP32 baseline comparison")
    regression = float(baseline["current_vs_baseline_percent"])
    receipt = {
        "schema": "cellerator.fp64.cost.v1",
        "source_revision": run(["git", "rev-parse", "HEAD"], cwd=args.repository).stdout.strip(),
        "benchmark_source_sha256": hashlib.sha256((Path(__file__).with_name("cost_bench.cu")).read_bytes()).hexdigest(),
        "driver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **metadata,
        "baseline_library": str(args.baseline_library.resolve()),
        "baseline_library_sidecar": str(sidecar.resolve()),
        "run_command": [str(args.binary.resolve()), "--baseline-library", str(args.baseline_library.resolve())],
        "measurements": records,
        "fp32_direct_regression_over_5_percent": regression > 5.0,
        "fp32_direct_regression_percent": regression,
        "limits": [
            "Each timed phase has 20 warmups and 200 measured repetitions; end-to-end repeats cold preparation, publication, forward and transpose apply, elementwise work, download, and teardown.",
            "Provider temporary device bytes are zero for measured apply/elementwise; this excludes allocator internals and host staging.",
            "The single workload is a representative sparse CSR case, not a broad performance claim.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "measurements": len(records),
                      "fp32_direct_regression_percent": regression,
                      "fp32_direct_regression_over_5_percent": regression > 5.0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
