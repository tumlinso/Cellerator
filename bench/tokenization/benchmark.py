#!/usr/bin/env python3
"""Small, explicit-device tokenizer comparison driver.

CPU is the safe default. CUDA is used only when ``--device cuda`` is supplied
inside a Device Scheduler job that has already granted that device. Results are
JSONL and include setup, input transfer, resident operations, and append setup
separately. This script makes no biological-performance claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch
from cellerator import tokenization as tok


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_baseline(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        stats = torch.cuda.memory_stats(device)
        return {
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "allocation_counter": int(stats.get("allocation.all.allocated", 0)),
        }
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(usage * (1024 if sys.platform.startswith("linux") else 1))
    return {"process_rss_highwater_bytes": rss_bytes}


def memory_result(device: torch.device, baseline: dict[str, int]) -> dict[str, int]:
    if device.type == "cuda":
        stats = torch.cuda.memory_stats(device)
        return {
            **baseline,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocation_counter_delta": int(stats.get("allocation.all.allocated", 0))
            - baseline["allocation_counter"],
        }
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        **baseline,
        "process_rss_highwater_bytes": int(
            usage * (1024 if sys.platform.startswith("linux") else 1)
        ),
        "rss_scope": "process_highwater_coarse_not_phase_precise",
    }


def timed(fn, device: torch.device, repeats: int) -> dict[str, object]:
    fn()
    synchronize(device)
    baseline = memory_baseline(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    memory = memory_result(device, baseline)
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "repeats": repeats,
        **memory,
    }


def numpy_rank(values: np.ndarray, features: np.ndarray, tokens: dict[int, int], k: int):
    rows = []
    for row in values:
        indices = np.flatnonzero(row != 0)
        order = np.lexsort((features[indices], -row[indices]))
        rows.append([tokens[int(features[j])] for j in indices[order[:k]]])
    return rows


def pack_ranked(row_ids: torch.Tensor, token_ids: torch.Tensor,
                n_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(row_ids, minlength=n_rows)
    offsets = torch.cat((counts.new_zeros((1,)), counts.cumsum(0)))
    return token_ids, offsets


def aten_dense_rank(values: torch.Tensor, features: torch.Tensor,
                    token_by_feature: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable ATen rank and ragged pack; ties use ascending feature identity."""
    identity_order = torch.argsort(features, stable=True)
    ordered_features = features.index_select(0, identity_order)
    ordered_tokens = token_by_feature.index_select(0, identity_order)
    ordered_values = values.index_select(1, identity_order)
    eligible = ordered_values != 0
    sortable = ordered_values.masked_fill(~eligible, -torch.inf)
    order = torch.argsort(sortable, dim=1, descending=True, stable=True)
    order = order[:, :min(k, values.shape[1])]
    selected = torch.gather(eligible, 1, order)
    selected_tokens = ordered_tokens.expand(values.shape[0], -1).gather(1, order)
    # Reading feature identities here keeps the reference explicit about the
    # axis mapping instead of relying on current column positions.
    _ = ordered_features.expand(values.shape[0], -1).gather(1, order)
    lengths = selected.sum(1, dtype=torch.int64)
    row_ids = torch.arange(values.shape[0], device=values.device).repeat_interleave(lengths)
    return pack_ranked(row_ids, selected_tokens[selected], values.shape[0])


def aten_csr_rank(values: torch.Tensor, features: torch.Tensor,
                  token_by_feature: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable segmented CSR rank without densifying observations."""
    crow = values.crow_indices()
    columns = values.col_indices()
    stored = values.values()
    row_lengths = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(
        torch.arange(values.shape[0], device=values.device), row_lengths,
        output_size=stored.numel(),
    )
    nonzero = stored != 0
    rows, columns, stored = rows[nonzero], columns[nonzero], stored[nonzero]
    identities = features.index_select(0, columns)
    tokens = token_by_feature.index_select(0, columns)

    # Stable least-significant-to-most-significant sorting gives row asc,
    # score desc, feature identity asc ordering with deterministic ties.
    order = torch.argsort(identities, stable=True)
    rows, stored, tokens = rows[order], stored[order], tokens[order]
    order = torch.argsort(stored, descending=True, stable=True)
    rows, tokens = rows[order], tokens[order]
    order = torch.argsort(rows, stable=True)
    rows, tokens = rows[order], tokens[order]

    counts = torch.bincount(rows, minlength=values.shape[0])
    starts = torch.repeat_interleave(
        counts.cumsum(0) - counts, counts, output_size=rows.numel()
    )
    keep = torch.arange(rows.numel(), device=rows.device) - starts < k
    return pack_ranked(rows[keep], tokens[keep], values.shape[0])


def packed_to_rows(packed: tuple[torch.Tensor, torch.Tensor]) -> list[list[int]]:
    token_ids, offsets = packed
    ids, offsets = token_ids.to("cpu").tolist(), offsets.to("cpu").tolist()
    return [ids[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def print_record(record: dict) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


def run_case(device: torch.device, rows: int, columns: int, density: float,
             repeats: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values_np = np.zeros((rows, columns), dtype=np.float32)
    present = rng.random((rows, columns)) < density
    values_np[present] = rng.lognormal(mean=0.0, sigma=1.0, size=int(present.sum())).astype(np.float32)
    features_np = rng.permutation(np.arange(100_000, 100_000 + columns, dtype=np.int64))
    tokens = {int(feature): i for i, feature in enumerate(features_np.tolist())}
    values_cpu = torch.from_numpy(values_np)
    features_cpu = torch.from_numpy(features_np)
    vocab_cpu = tok.make_vocabulary(
        torch.from_numpy(np.sort(features_np)),
        torch.tensor([tokens[int(x)] for x in np.sort(features_np)], dtype=torch.int64),
    )
    token_by_feature_cpu = torch.tensor(
        [tokens[int(feature)] for feature in features_np], dtype=torch.int64
    )

    transfer_memory_before = memory_baseline(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    transfer_start = time.perf_counter()
    values = values_cpu.to(device)
    features = features_cpu.to(device)
    token_by_feature = token_by_feature_cpu.to(device)
    vocab = tok.make_vocabulary(vocab_cpu.identities.to(device), vocab_cpu.token_ids.to(device))
    dense = values
    sparse = dense.to_sparse_csr()
    synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_start
    transfer_memory = memory_result(device, transfer_memory_before)

    oracle_start = time.perf_counter()
    expected = numpy_rank(values_np, features_np, tokens, 256)
    oracle_seconds = time.perf_counter() - oracle_start
    options = tok.RankedFeatureOptions()
    options.max_per_row = 256

    dense_out = tok.ranked_feature(dense, features, vocab, options)
    got = dense_out.token_ids.to("cpu")
    offsets = dense_out.offsets.to("cpu")
    got_rows = [got[offsets[i]:offsets[i + 1]].tolist() for i in range(rows)]
    if got_rows != expected:
        raise AssertionError("ranked_feature output differs from the NumPy reference")
    dense_reference = aten_dense_rank(dense, features, token_by_feature, 256)
    csr_reference = aten_csr_rank(sparse, features, token_by_feature, 256)
    if packed_to_rows(dense_reference) != expected:
        raise AssertionError("dense ATen composition differs from the NumPy reference")
    if packed_to_rows(csr_reference) != expected:
        raise AssertionError("CSR ATen composition differs from the NumPy reference")

    common = {
        "device": str(device), "rows": rows, "features": columns,
        "density": density, "nnz": int(present.sum()), "top_k": 256,
        "storage_dtype": "float32", "seed": seed,
    }
    print_record({**common, "phase": "input_transfer_and_sparse_projection",
                  "seconds": transfer_seconds, **transfer_memory})
    print_record({**common, "phase": "numpy_reference",
                  **timed(lambda: numpy_rank(values_np, features_np, tokens, 256),
                          torch.device("cpu"), repeats),
                  "first_run_seconds": oracle_seconds})
    print_record({**common, "phase": "ranked_feature_dense",
                  **timed(lambda: tok.ranked_feature(dense, features, vocab, options),
                          device, repeats)})
    print_record({**common, "phase": "aten_composition_dense_rank_pack",
                  **timed(lambda: aten_dense_rank(dense, features, token_by_feature, 256),
                          device, repeats)})
    print_record({**common, "phase": "ranked_feature_csr",
                  **timed(lambda: tok.ranked_feature(sparse, features, vocab, options),
                          device, repeats)})
    print_record({**common, "phase": "aten_composition_csr_segmented_rank_pack",
                  **timed(lambda: aten_csr_rank(sparse, features, token_by_feature, 256),
                          device, repeats)})

    csr = dense.to_sparse_csr()
    print_record({**common, "phase": "feature_median_csr",
                  **timed(lambda: tok.nonzero_feature_median([csr], 256),
                          device, repeats)})


def run_append(device: torch.device, rows: int, sequence_length: int,
               events_per_append: int, repeats: int, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    capacities = [sequence_length + events_per_append * repeats + 1] * rows
    ids = torch.randint(0, 10_000, (rows, sequence_length), generator=generator)
    lengths = torch.full((rows,), sequence_length, dtype=torch.int64)
    feature_offsets = torch.cat((torch.tensor([0]), lengths.cumsum(0)))
    updates = torch.randint(0, 10_000, (rows, events_per_append), generator=generator)
    update_lengths = torch.full((rows,), events_per_append, dtype=torch.int64)
    update_offsets = torch.cat((torch.tensor([0]), update_lengths.cumsum(0)))
    ids_dev = ids.reshape(-1).to(device)
    updates_dev = updates.reshape(-1).to(device)
    buffer_examples = {"tick": torch.empty((0,), dtype=torch.int64, device=device)}
    selected = torch.arange(rows, dtype=torch.int64, device=device)
    prefix_tick = torch.zeros(ids_dev.numel(), dtype=torch.int64, device=device)
    update_tick = torch.full((updates_dev.numel(),), sequence_length,
                             dtype=torch.int64, device=device)
    prefix = tok.TokenBatch(ids_dev, feature_offsets.to(device), {"tick": prefix_tick})
    update_batch = tok.TokenBatch(
        updates_dev, update_offsets.to(device), {"tick": update_tick}
    )
    append_status = torch.empty((rows,), dtype=torch.bool, device=device)

    def seeded_buffer():
        item = tok.SequenceBuffer.create(capacities, buffer_examples, str(device))
        item.append_checked(selected, prefix)
        return item

    setup_memory_before = memory_baseline(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    buffer = seeded_buffer()
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    setup_memory = memory_result(device, setup_memory_before)

    def append_sample():
        sample_buffer = seeded_buffer()
        synchronize(device)
        memory_before = memory_baseline(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = None
        for _ in range(repeats):
            result = sample_buffer.append_into(
                selected, update_batch, status_buffer=append_status
            )
        synchronize(device)
        elapsed = time.perf_counter() - start
        if result is None:
            raise AssertionError("append loop did not execute")
        measured_memory = memory_result(device, memory_before)
        return elapsed, measured_memory, result.succeeded

    append_sample()  # warmup on a fresh buffer
    append_samples = [append_sample() for _ in range(5)]
    append_timing = statistics.median(sample[0] for sample in append_samples)
    append_memory = append_samples[-1][1]
    if not bool(append_samples[-1][2].all().item()):
        raise AssertionError("unexpected sequence buffer overflow")

    # Equivalent fixed-capacity ATen storage baseline. Row-major storage and
    # per-step destination indices are prepared once; each append writes only
    # the new event payload and increments lengths, without rebuilding prior
    # histories. The buffer and baseline use the same token and tick batches.
    baseline_capacity = capacities[0]

    def validate_baseline_attributes() -> None:
        if update_tick.dtype != torch.int64 or update_tick.numel() != updates_dev.numel():
            raise AssertionError("ATen append baseline tick schema does not match token payload")
        if prefix_tick.dtype != torch.int64 or prefix_tick.numel() != ids_dev.numel():
            raise AssertionError("ATen append baseline seed tick schema does not match token payload")

    def aten_seeded_storage():
        token_storage = torch.empty(rows * baseline_capacity, dtype=ids_dev.dtype,
                                    device=device)
        tick_storage = torch.empty(rows * baseline_capacity, dtype=update_tick.dtype,
                                   device=device)
        token_storage.view(rows, baseline_capacity)[:, :sequence_length].copy_(
            ids.view(rows, sequence_length).to(device)
        )
        tick_storage.view(rows, baseline_capacity)[:, :sequence_length].copy_(
            prefix_tick.view(rows, sequence_length)
        )
        lengths_storage = torch.full((rows,), sequence_length, dtype=torch.int64,
                                     device=device)
        return token_storage, tick_storage, lengths_storage

    baseline_setup_memory_before = memory_baseline(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_setup_start = time.perf_counter()
    validate_baseline_attributes()
    baseline_destinations = torch.stack([
        (torch.arange(rows, device=device)[:, None] * baseline_capacity
         + torch.arange(sequence_length + step * events_per_append,
                        sequence_length + (step + 1) * events_per_append,
                        device=device)[None, :]).reshape(-1)
        for step in range(repeats)
    ])
    baseline_tokens, baseline_ticks, baseline_lengths = aten_seeded_storage()
    synchronize(device)
    baseline_setup_seconds = time.perf_counter() - baseline_setup_start
    baseline_setup_memory = memory_result(device, baseline_setup_memory_before)

    def aten_append_sample():
        token_storage, tick_storage, lengths_storage = aten_seeded_storage()
        synchronize(device)
        memory_before = memory_baseline(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for step in range(repeats):
            token_storage.index_copy_(0, baseline_destinations[step], updates_dev)
            tick_storage.index_copy_(0, baseline_destinations[step], update_tick)
            lengths_storage.add_(events_per_append)
        synchronize(device)
        memory_after = memory_result(device, memory_before)
        return (time.perf_counter() - start, token_storage, tick_storage,
                lengths_storage, memory_after)

    aten_append_sample()  # warmup
    aten_samples = [aten_append_sample() for _ in range(5)]
    aten_timing = statistics.median(sample[0] for sample in aten_samples)
    aten_tokens, aten_ticks, aten_lengths, aten_memory = aten_samples[-1][1:]
    reference_buffer = seeded_buffer()
    for _ in range(repeats):
        reference_buffer.append_into(selected, update_batch)
    buffer_tokens = reference_buffer.compact().token_ids
    aten_compact = torch.cat([
        aten_tokens.view(rows, baseline_capacity)[row, :int(aten_lengths[row].item())]
        for row in range(rows)
    ])
    torch.testing.assert_close(aten_compact, buffer_tokens)
    buffer_ticks = reference_buffer.compact().attributes["tick"]
    aten_compact_ticks = torch.cat([
        aten_ticks.view(rows, baseline_capacity)[row, :int(aten_lengths[row].item())]
        for row in range(rows)
    ])
    torch.testing.assert_close(aten_compact_ticks, buffer_ticks)
    print_record({
        "device": str(device), "phase": "sequence_buffer_setup_and_seed",
        "rows": rows, "sequence_length": sequence_length,
        "events_per_append": events_per_append, "setup_seconds": setup_seconds,
        "reserved_tokens": int(sum(capacities)), "seed": seed,
        **setup_memory,
    })
    print_record({
        "device": str(device), "phase": "append_steady_state",
        "rows": rows, "sequence_length": sequence_length,
        "events_per_append": events_per_append, "append_calls": repeats,
        "seconds_per_call": append_timing / repeats,
        "total_seconds": append_timing,
        "repeats": len(append_samples), "seed": seed,
        **append_memory,
    })
    print_record({
        "device": str(device), "phase": "aten_preallocated_append_setup_and_seed",
        "rows": rows, "sequence_length": sequence_length,
        "events_per_append": events_per_append,
        "setup_seconds": baseline_setup_seconds,
        "reserved_tokens": rows * baseline_capacity, "seed": seed,
        **baseline_setup_memory,
    })
    print_record({
        "device": str(device), "phase": "aten_preallocated_append_steady_state",
        "rows": rows, "sequence_length": sequence_length,
        "events_per_append": events_per_append, "append_calls": repeats,
        "seconds_per_call": aten_timing / repeats,
        "total_seconds": aten_timing, "repeats": len(aten_samples), "seed": seed,
        **aten_memory,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--features", type=int, default=20_000)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.10])
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--events-per-append", type=int, default=8)
    parser.add_argument("--append-calls", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested explicitly but unavailable in this interpreter")
    if args.rows <= 0 or args.features <= 0 or args.repeats <= 0:
        parser.error("rows, features, and repeats must be positive")
    if any(not (0.0 < density <= 1.0) for density in args.densities):
        parser.error("each density must be in (0, 1]")

    print_record({
        "phase": "environment", "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "platform": platform.platform(),
        "threads": torch.get_num_threads(), "seed": args.seed,
    })
    for density in args.densities:
        run_case(device, args.rows, args.features, density, args.repeats, args.seed)
    run_append(device, args.rows, args.sequence_length,
               args.events_per_append, args.append_calls, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
