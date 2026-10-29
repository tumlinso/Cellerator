#!/usr/bin/env python3
"""Create a v13-pinned copy of the frozen v12 packed/SIMT KDA contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
BASE_TASK = REPO / "bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027/kda_task_v12.json"
PACKAGE = REPO / "build-sparse-support/evidence/torch-v13-owner-validation/package"
IDENTITY = REPO / "build-sparse-support/evidence/torch-v13-owner-validation/identity.json"
EXTENSION = PACKAGE / "cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so"
EXTENSION_HASH = hashlib.sha256(EXTENSION.read_bytes()).hexdigest()
IDENTITY_HASH = hashlib.sha256(IDENTITY.read_bytes()).hexdigest()
FIXTURE_ROOT = Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    task = json.loads(BASE_TASK.read_text())
    task["task_id"] = "cellerator-scaled-sparse-packed-vs-simt-20261028-v13-owner-validation"
    task["toolchain"]["runtime"] = "frozen v13 owner-validation Torch extension; Volta kernels; Torch CUDA runtime 12.6"
    task["commands"]["benchmark"]["argv"][1] = str(REPO / "bench/sparse_support/kda_adapter.py")
    task["commands"]["validate"]["argv"][1] = str(REPO / "bench/sparse_support/kda_adapter.py")
    task["commands"]["build"]["argv"][1] = str(REPO / "bench/sparse_support/kda_adapter.py")
    task["probe"]["argv"][1] = str(REPO / "bench/sparse_support/kda_adapter.py")

    for case in task["cases"]:
        if "fixture_root" in case["input"]:
            case["input"]["fixture_root"] = str(FIXTURE_ROOT)

    trusted = []
    for item in task["trusted_inputs"]:
        if item["id"] == "frozen_torch_v12_identity":
            item["id"] = "frozen_torch_v13_identity"
            item.update(path=str(IDENTITY), sha256=IDENTITY_HASH)
        elif item["id"] == "frozen_v12_sparse_support_python":
            item["id"] = "frozen_v13_sparse_support_python"
            item.update(path=str(PACKAGE / "cellerator/torch/sparse_support.py"))
        elif item["id"] == "frozen_v12_mechanism_python":
            item["id"] = "frozen_v13_mechanism_python"
            item.update(path=str(PACKAGE / "cellerator/torch/mechanism.py"))
        elif item["id"] == "frozen_v12_cuda_training_test":
            item["id"] = "frozen_v13_cuda_training_test"
            item.update(path=str(IDENTITY.parent / "tests/test_cuda_training.py"))
        elif item["id"] == "frozen_v12_torch_extension":
            item["id"] = "frozen_v13_torch_extension"
            item.update(path=str(EXTENSION), sha256=EXTENSION_HASH)
        elif item["id"] == "large_ranked_v3_manifest":
            item.update(path=str(FIXTURE_ROOT / "large-ranked/manifest.json"))
        elif item["id"] == "large_events_v3_manifest":
            item.update(path=str(FIXTURE_ROOT / "large-events/manifest.json"))
        elif item["id"] == "scaled_fixture_generator":
            item.update(path=str(REPO / "bench/sparse_support/scaled_fixtures.py"))
        trusted.append(item)
    task["trusted_inputs"] = trusted
    write_json(OUT / "task.json", task)

    dependency = [{
        "name": "cellerator-torch-v13-owner-validation-extension",
        "revision": EXTENSION_HASH,
    }, {"name": "torch-cuda-runtime", "revision": "2.7.0+cu126"},
       {"name": "target", "revision": "sm_70"}]
    for mode, role, dispatch in (
        ("simt", "incumbent_control", "native SIMT forward and VJP"),
        ("packed", "measured_candidate", "CE-GEO N64 forward plus shared SIMT VJP"),
    ):
        candidate = OUT / "candidates" / mode
        candidate.mkdir(parents=True, exist_ok=True)
        write_json(candidate / "backend.json", {
            "backend": mode,
            "dispatch": dispatch,
            "frozen_extension_sha256": EXTENSION_HASH,
            "role": role,
        })
        write_json(candidate / "manifest.json", {
            "dependencies": dependency,
            "files": ["backend.json"],
            "schema_version": 1,
        })
    write_json(OUT / "package_identity.json", {
        "package_identity_sha256": IDENTITY_HASH,
        "extension_sha256": EXTENSION_HASH,
        "fixture_root": str(FIXTURE_ROOT),
    })


if __name__ == "__main__":
    main()
