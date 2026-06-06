---
slug: "baseplane-dna2-explicit-widths"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-06-05T14:09:21Z"
last_heartbeat_at: "2026-06-05T14:18:24Z"
last_reviewed_at: "2026-06-05T14:09:21Z"
stale_after_days: 3
objective: "Implement explicit-width dna2 packed, split-plane, and inline-plane representations in Baseplane"
---

# Current Objective

## Summary
Implement explicit-width dna2 packed, split-plane, and inline-plane representations in Baseplane

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Read `../Baseplane/AGENTS.md` before modifying sibling Baseplane.
- Use cuda guidance for device helper changes; no benchmark variant is required in this pass.
- Hard rename ambiguous dna2_word and dna2_planes to explicit-width public types; do not leave compatibility aliases.

## Planning Notes
_None recorded yet._

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda` - Low-level sequence representation change touches CUDA device helpers and should preserve optional CUDA builds.
- `todo-orchestrator` - Track this multi-file Baseplane API migration through validation.

## Useful Reference Files
- `../Baseplane/AGENTS.md` - Baseplane scope, CUDA policy, and style constraints.
- `../Baseplane/include/Baseplane/seq/dna2.cuh` - Public dna2 API and device-inline helpers.
- `../Baseplane/tests/seq/test_dna2.cpp` - CPU correctness coverage for representations.
- `../Baseplane/tests/seq/test_dna2_cuda.cu` - CUDA device helper coverage.

## Plan
_None recorded yet._

## Tasks
- [x] Update public dna2 representation structs and helpers to explicit-width names.
- [x] Update scalar, Highway, CUDA, bench, and docs call sites for hard rename.
- [x] Extend CPU and CUDA tests for word32, planes64, and inline-plane conversions/ops.
- [x] Build and run focused Baseplane validation targets.

## Blockers
_None recorded yet._

## Progress Notes
- Added explicit dna2_word32/64, dna2_planes32/64, and dna2_inlplane32/64 representations plus conversion, mismatch, exact-match, and reverse-complement helpers.
- Removed active ambiguous dna2_word and dna2_planes names from Baseplane call sites.
- Validation passed in sibling Baseplane: `cmake -S . -B build`; `cmake --build build --target baseplaneDna2Test baseplaneDna2CudaTest -j 4`; `./build/baseplaneDna2Test`; `./build/baseplaneDna2CudaTest`.

## Next Actions
- Edit dna2.cuh first, then update implementation and tests until rg finds no active dna2_word/dna2_planes ambiguous names.
- Commit Baseplane submodule changes first, then update the Cellerator umbrella pointer and ledger changes if desired.

## Done Criteria
- No active Baseplane code uses ambiguous dna2_word or dna2_planes names.
- baseplaneDna2Test passes.
- baseplaneDna2CudaTest passes when CUDA is enabled.
