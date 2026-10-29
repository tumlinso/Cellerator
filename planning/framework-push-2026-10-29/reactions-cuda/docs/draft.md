# Sparse reaction event CUDA draft and executable plan

Operation: apply caller-supplied ordered discrete reaction events to private
int64 count rows sharing a CSR reaction bank. Exact arithmetic, no sampling.
One thread owns a cell; validates a complete sparse delta before committing.
Sorted unique species per reaction permits linear validation with no scratch.
Independent cells run concurrently. Per-cell fixed-capacity change logs avoid
atomics; capacity failure occurs before event commit. An explicit optional
ordinal map connects physical species slots to incremental frontier sources.
Event index indirection supports event-major storage with ragged per-cell order.

Baseline/oracle: native ReactionNetwork::workspace().fire() and manually checked
counts; no hardware performance baseline or tuning claim. Warp-per-cell is a
possible later candidate for broad stoichiometry, deferred without measurements.
Risks: malformed device offsets, duplicates, overflow, rollback, frontier
capacity, caller aliasing/lifetime. Wrapper checks extents; kernel checks contents.

Steps: public borrowed-view contract, per-cell kernel and checked launch wrapper,
smoke fixture comparing ordered counts and rollback, SM70 compile/link only.
No GPU execution or benchmarking authorized in this assignment. Runtime and
sanitizer evidence remains pending; compile success is not correctness proof.

Commands: nvcc -std=c++17 -arch=sm_70 -Iinclude -c
src/modeling/cuda/reactions.cu -o /tmp/cellerator-reactions-cuda.o;
compile/link tests/framework/cuda/reactions_smoke.cu with runtime.cu/reactions.cu.
Target source delivery 18:45 UTC, 2026-10-29.
References consumed: CUDA SKILL.md, common/machine-aligned-design.md,
systems/native.md; KDA docs/agent-entry.md, docs/agent-flow.md,
prompts/basic-flow.md as installed 2026-10-29. Existing runtime.hh/frontier.hh
are the borrowed storage/stream and invalidation interfaces.
