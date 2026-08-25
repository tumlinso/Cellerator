#!/usr/bin/env python3
"""Run each CE-ARCH-92 trace/N cell in an isolated benchmark process."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--n", type=int, action="append", required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if any(width < 1 or width > 64 for width in options.n):
        raise SystemExit("N must be inside the native 1..64 capability range")
    if options.warmups < 1 or options.repeats < 3:
        raise SystemExit("timing policy requires warmups >= 1 and repeats >= 3")

    chunks: list[bytes] = []
    with tempfile.TemporaryDirectory(prefix="ce-arch-92-") as directory:
        scratch = Path(directory)
        for trace_index, trace in enumerate(options.trace):
            for width in options.n:
                result = scratch / f"trace-{trace_index}-n-{width}.jsonl"
                subprocess.run(
                    [
                        str(options.binary), "--trace", str(trace),
                        "--output", str(result), "--n", str(width),
                        "--warmups", str(options.warmups),
                        "--repeats", str(options.repeats),
                    ],
                    check=True,
                )
                chunks.append(result.read_bytes())
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_bytes(b"".join(chunks))


if __name__ == "__main__":
    main()
