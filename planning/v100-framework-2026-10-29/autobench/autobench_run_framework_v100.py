#!/usr/bin/env python3
"""Run snapshot-bound V100 smoke, sanitizer, or timing packets for Autobench."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CASES = {
    "runtime": "cellerator_framework_cuda_runtime_smoke",
    "dispatch": "cellerator_framework_cuda_dispatch_smoke",
    "quantization": "cellerator_framework_cuda_quantization_smoke",
    "memory": "cellerator_framework_cuda_memory_smoke",
    "frontier": "cellerator_framework_cuda_frontier_smoke",
    "attention": "cellerator_framework_cuda_attention_smoke",
    "grouped_linear": "cellerator_framework_cuda_grouped_linear_smoke",
    "packed_embedding": "cellerator_framework_cuda_packed_embedding_smoke",
    "reactions": "cellerator_framework_cuda_reactions_smoke",
    "regulatory_gates": "cellerator_framework_cuda_regulatory_gates_smoke",
    "parameter_transfer": "cellerator_framework_cuda_parameter_transfer_smoke",
}
BENCHES = {
    "attention": ("cellerator_framework_cuda_attention_smoke", ["--benchmark"]),
    "grouped_linear": ("cellerator_v100_grouped_linear_benchmark", ["--gpu"]),
    "packed_embedding": ("cellerator_framework_cuda_packed_embedding_smoke", ["--performance"]),
    "dispatch": ("cellerator_v100_dispatch_benchmark", ["1", "8", "32", "33", "256", "2048"]),
    "memory": ("cellerator_v100_memory_benchmark", []),
    "quantization": ("cellerator_v100_quantization_benchmark", []),
    "regulatory": ("cellerator_v100_regulatory_benchmark", ["4099", "512", "128", "10", "100"]),
    "frontier": ("cellerator_frontier_candidate_benchmark", []),
    "packed_embedding_candidate": ("cellerator_v100_packed_embedding_candidate_smoke", ["--performance"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> tuple[int, str, float]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, check=False)
    return result.returncode, result.stdout, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("correctness", "benchmark"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--python", default="/home/tumlinson/Software/miniconda3/bin/python")
    parser.add_argument("--tool", choices=("none", "memcheck", "racecheck", "initcheck", "synccheck"), default="none")
    parser.add_argument("--sanitizer", default="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer")
    args = parser.parse_args()
    source, build, output = args.source.resolve(strict=True), args.build.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(source / "python"), "PYTHONDONTWRITEBYTECODE": "1",
                "CELLERATOR_TEST_CUDA": "1", "CELLERATOR_TEST_CUDA_DEVICE": "0",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    records = []
    for case in args.cases:
        if args.mode == "correctness" and case == "bindings":
            binary = build / "_modeling_cuda.cpython-313-x86_64-linux-gnu.so"
            test_file = source / "tests/framework/cuda/test_runtime_binding.py"
            code = (
                "import importlib.util,sys,unittest; "
                f"sys.path.insert(0,{str(source / 'python')!r}); "
                "import cellerator; "
                f"cellerator.__path__.append({str(build)!r}); "
                f"spec=importlib.util.spec_from_file_location('test_runtime_binding',{str(test_file)!r}); "
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "suite=unittest.defaultTestLoader.loadTestsFromModule(module); "
                "result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"
            )
            command = [args.python, "-c", code]
        elif args.mode == "correctness":
            if case not in CASES:
                raise ValueError(f"unknown correctness case: {case}")
            binary = build / CASES[case]
            args_by_case = {"attention": ["--device"], "grouped_linear": ["--gpu"], "reactions": ["--device", "0"]}
            command = [str(binary), *args_by_case.get(case, [])]
        else:
            if case not in BENCHES:
                raise ValueError(f"unknown benchmark case: {case}")
            target, workload = BENCHES[case]
            if case == "frontier":
                binary = build.parent / "frontier-candidate" / target
            else:
                binary = build / target
            if case == "dispatch":
                for feature in workload:
                    records.append(run_one(case, binary, feature, args, source, output, env))
                continue
            command = [str(binary), *workload]

        if not binary.is_file():
            raise FileNotFoundError(binary)
        if args.tool != "none":
            command = [args.sanitizer, "--tool", args.tool, "--error-exitcode", "99", *command]
        result, log, elapsed = run(command, cwd=source, env=env, timeout=240)
        log_path = output / f"{case}-{args.mode}-{args.tool}.log"
        log_path.write_text(log, encoding="utf-8")
        records.append({"case": case, "mode": args.mode, "tool": args.tool,
                        "argv": command, "returncode": result, "wall_seconds": elapsed,
                        "binary_sha256": sha256(binary), "log": log_path.name})
        print(log, end="", flush=True)
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    (output / "results.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(any(record["returncode"] for record in records))


def run_one(case: str, binary: Path, workload: str, args: argparse.Namespace,
            source: Path, output: Path, env: dict[str, str]) -> dict[str, object]:
    command = [str(binary), workload]
    result, log, elapsed = run(command, cwd=source, env=env, timeout=240)
    log_path = output / f"{case}-{workload}.log"
    log_path.write_text(log, encoding="utf-8")
    record = {"case": case, "mode": args.mode, "tool": args.tool,
              "argv": command, "returncode": result, "wall_seconds": elapsed,
              "binary_sha256": sha256(binary), "log": log_path.name}
    print(log, end="", flush=True)
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


if __name__ == "__main__":
    raise SystemExit(main())
