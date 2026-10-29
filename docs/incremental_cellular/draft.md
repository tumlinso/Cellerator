# Incremental frontier implementation draft

Scope: cache pure interaction callbacks from immutable source reads; invalidate
from declared role/entity dependencies and parameter/context epochs. Build CUDA
frontiers from source CSR postings, not dense pairs. Stable relation IDs and
participant multiplicity remain separate from dense execution ordinals.

CPU oracle: fresh full recomputation after localized state change. Required
checks: reduced callback count with identical outputs, pending ineligible work,
epoch changes, duplicate updates, repeated incidences, failed callback rollback.
CUDA oracle: sorted unique union of selected host CSR rows; cover duplicates,
empty input, capacity overflow, and invalid source/posting indices.

Candidate: one warp per changed source, atomic claim per interaction, compact
atomic output reservation. Scratch claims are reset in a separate stream-ordered
pass. This pays O(interactions) scratch clearing but visits only selected posting
rows and performs no dense pair scan. A generation-tagged persistent workspace
could avoid clearing but adds generation/lifetime complexity and is deferred.
No measured performance claim; callback count is an algorithmic check only.

Implement a C++17 header cache and CSR preparation, then CUDA borrowed-view API
with explicit stream, caller-owned storage and asynchronous status. Compile the
CPU test against existing sparse support/indexing; compile CUDA for sm_70 using
installed nvcc. No GPU execution, profiling, benchmark, or installation is
authorized in this workstream. Device correctness and sanitizers remain pending.

Guidance read: CUDA SKILL.md, Volta router, native system and kernel mechanics;
KDA entry/basic flow/agent flow at 277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc;
KernelWiki skill and hw-volta-integer-routing at
542ed702ac662422d55b19186562b29abbb5da29.
