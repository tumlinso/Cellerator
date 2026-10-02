#!/usr/bin/env python3
"""Small dependency-free learning-to-LOP3 fixture, not a training framework.
Learns eight independent Bernoulli logits from a synthetic Boolean teacher.
This only proves the path from continuous parameters to an instruction LUT.
No genome, biological target, or held-out generalization claim is involved.
"""
import argparse
import json
import math
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be in [1, 10000]")
    logits = [0.0] * 8
    target = [((k >> 2) & 1) ^ (((k >> 1) & 1) & (k & 1)) for k in range(8)]
    for _ in range(args.steps):
        for k, y in enumerate(target):
            p = 1.0 / (1.0 + math.exp(-logits[k]))
            logits[k] -= 0.5 * (p-y)
    lut = sum((z > 0.0) << k for k, z in enumerate(logits))
    expected = sum(y << k for k, y in enumerate(target))
    if lut != expected:
        raise RuntimeError("hardened LUT does not match the synthetic teacher")
    result = {"fixture": "synthetic_truth_table", "logits": logits,
              "lop3_immediate": f"0x{lut:02x}", "teacher_lut": f"0x{expected:02x}",
              "truth_table_agreement": lut == expected, "biological_validation": False,
              "generalization_evaluated": False}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")

if __name__ == "__main__":
    main()
