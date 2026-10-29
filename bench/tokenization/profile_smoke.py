#!/usr/bin/env python3
"""Finite, small CUDA tokenization workload annotated for Nsight Systems.

Run only inside a Device Scheduler job that explicitly grants a GPU. Example:

  nsys profile -t cuda,nvtx --sample=none --stats=true \
    python bench/tokenization/profile_smoke.py --steps 64

This is a profiler smoke workload, not a benchmark or biological validation.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from cellerator import tokenization as tok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--events-per-step", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.sequences < 1 or args.events_per_step < 1:
        parser.error("steps, sequences and events-per-step must be positive")
    if not torch.cuda.is_initialized():
        # Initialize only after execution was explicitly admitted by the caller.
        torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    sequences = args.sequences
    vocab_width = 256
    vocabulary_ids = torch.arange(vocab_width, dtype=torch.int64, device=device)
    vocab = tok.make_vocabulary(vocabulary_ids, vocabulary_ids)
    event_ids = torch.arange(sequences, dtype=torch.int64, device=device) % vocab_width
    event_offsets = torch.arange(sequences + 1, dtype=torch.int64, device=device)
    ticks = torch.zeros(sequences, dtype=torch.int64, device=device)
    selected = torch.arange(sequences, dtype=torch.int64, device=device)

    # Exercise the same public operations used by a caller, with explicit NVTX
    # ranges to make event encoding, feature ranking and storage visible.
    with torch.cuda.nvtx.range("cellerator_tokenization_setup"):
        buffer = tok.SequenceBuffer.create(
            [args.steps * args.events_per_step + 1] * sequences,
            {"tick": torch.zeros(1, dtype=torch.int64, device=device)},
            str(device),
        )
        initial = tok.event_stream(
            event_ids, event_offsets, vocab, {"tick": ticks}
        )
        seed_status = buffer.append_into(selected, initial)
        dense = torch.rand((sequences, vocab_width), device=device)
        features = vocabulary_ids

    # Keep a bounded ranking capture between the setup and append ranges.
    with torch.cuda.nvtx.range("cellerator_ranked_feature"):
        ranked = tok.ranked_feature(dense, features, vocab)

    repeated_ids = event_ids.repeat_interleave(args.events_per_step)
    step_offsets = torch.arange(
        0, (sequences + 1) * args.events_per_step,
        args.events_per_step, device=device, dtype=torch.int64,
    )
    step_ticks = torch.full((repeated_ids.numel(),), 1,
                            dtype=torch.int64, device=device)
    append_batch = tok.TokenBatch(repeated_ids, step_offsets, {"tick": step_ticks})

    started = time.perf_counter()
    with torch.cuda.nvtx.range("cellerator_sequence_append_repeated"):
        last_status = None
        for _ in range(args.steps):
            last_status = buffer.append_into(selected, append_batch)
    # One synchronization after the complete append series; do not add a host
    # readback to each measured step.
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if last_status is None or not bool(last_status.succeeded.all().item()):
        raise RuntimeError("profile smoke sequence buffer capacity was exceeded")
    if not bool(seed_status.succeeded.all().item()):
        raise RuntimeError("profile smoke seed append unexpectedly overflowed")
    expected_minimum = args.steps * args.events_per_step + 1
    if not bool((buffer.lengths >= expected_minimum).all().item()):
        raise RuntimeError("profile smoke append lengths did not advance")
    print(json.dumps({
        "device": str(device), "steps": args.steps, "sequences": sequences,
        "events_per_step": args.events_per_step,
        "append_wall_seconds": elapsed,
        "final_lengths": buffer.lengths.cpu().tolist(),
        "ranked_token_count": int(ranked.token_ids.numel()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
