#!/usr/bin/env python3
"""Verify and flatten the local CE-LIVE PBMC3K NPZ for the CUDA test."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cellerator.ce-live.quantitative-fixture.v1":
        raise SystemExit("unexpected CE-LIVE fixture manifest schema")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        indptr = np.ascontiguousarray(fixture["indptr"], dtype="<u8")
        indices = np.ascontiguousarray(fixture["indices"], dtype="<u4")
        generation_1 = np.ascontiguousarray(
            fixture["generation_1_values"], dtype="<f4")
        generation_2 = np.ascontiguousarray(
            fixture["generation_2_values"], dtype="<f4")

    csr = manifest["extracted_csr"]
    rows, columns = map(int, csr["shape"])
    nnz = int(csr["nnz"])
    expected = {
        "indptr": csr["indptr_sha256"],
        "indices": csr["indices_sha256"],
        "generation_1": manifest["generations"][0]["values_sha256"],
        "generation_2": manifest["generations"][1]["values_sha256"],
    }
    actual = {
        "indptr": digest(indptr),
        "indices": digest(indices),
        "generation_1": digest(generation_1),
        "generation_2": digest(generation_2),
    }
    if actual != expected:
        raise SystemExit(f"fixture checksum mismatch: {actual}")
    if (indptr.shape != (rows + 1,) or indices.shape != (nnz,)
            or generation_1.shape != (nnz,)
            or generation_2.shape != (nnz,)
            or int(indptr[0]) != 0 or int(indptr[-1]) != nnz):
        raise SystemExit("fixture shape or CSR terminal offsets are invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"CELIVE31")
        stream.write(struct.pack("<IIIIQ", 1, rows, columns, 0, nnz))
        stream.write(indptr.tobytes(order="C"))
        stream.write(indices.tobytes(order="C"))
        stream.write(generation_1.tobytes(order="C"))
        stream.write(generation_2.tobytes(order="C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
