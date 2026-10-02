# What was actually checked

This delivery is a research/bootstrap package with source prototypes. It is **not an installed Cellerator feature or a performance-qualified GPU implementation**.

## Executed locally

- Fifteen NumPy algebra checks passed: tensorized-map equivalence; patch JVP/VJP and finite differences; independent actor-local coordinate changes; zero-value/nonzero-response products; repeated product arguments; a bijection of the documented quad-MMA output coordinates; supplied linear refactoring; invariant-manifold quotient and replica pullback; epoch/tape protection; live-gradient capacity recycling; nonmerging footprint placement; last-transmitted delta behavior; low-rank implicit solve; quadratic-field responses; and the linear-flow semigroup witness.
- Ordinary Torch CPU forward/backward checks passed for the patch, actor-port, regrowth and exact linear-flow references.
- Two tiny synthetic rollout-fitting examples executed for eight optimizer steps each. Both losses decreased in this run. These are engineering smoke examples, not biological or generalization evidence, and not performance measurements.
- The C++17 state-layout seed compiled under g++ with `-Wall -Wextra -Werror` and its executable passed.
- Both plan files passed local ID/reference/DAG/lane-order consistency checks. These local checks do not replace native authority validation.
- Copy-helper smoke checks use temporary synthetic git repositories only; see `validation/bootstrap_smoke.txt` for the final result. No connected repository was altered.

Logs and exact local Python/NumPy/Torch versions are in `validation/`. The local Torch installation is CPU-only. The prototypes do not require changing the project's existing compatible CUDA/Torch environment.

## Native plan preview

**GlassHelix:** Project Control returned `valid=true` at Todo revision260, source `34782c48d3d89465af0b635eb78b56d872c212ac`; five additive tasks, no modifications, no dependency/scope/interface errors, mutation guard unchanged. The receipt is summarized in `validation/plan-preview-receipts.json` with the package-file hash and the tool's separate normalized plan digest.

**Cellerator:** The target-specific preview failed with `internal_error` and `bounded_read_failed`. No successful CE native validation is claimed. The file follows the inspected schema-v3 structure, but successful CE-target validation is mandatory before import.

Neither plan was applied. No tasks, claims, queues, interfaces or gates were mutated by this delivery. The previews are dated observations and must be refreshed before adoption.

## Not executed

There is no `nvcc` and no CUDA device in the local execution container. None of the three CUDA source prototypes was compiled, run, profiled or sanitized here. The supplied quad-MMA device test is unrun; the CPU coordinate-bijection check does not establish that its PTX binding is correct on hardware.

No remote Cellerator/GlassHelix build, installation, GPU test, source edit or benchmark was performed. Native patch/port backward adapters, general nonlinear extraction, scheduler integration, and several optional mechanism cards remain work for the bootstrap tasks. References using float64 or ordinary Torch are not proof of bitwise parity with half Tensor-Core execution.

## Deliberately limited validation budget

The implementing tasks ask for a tiny same-policy numerical check, requested gradients, legal memory/identity admission and a callable consumer. They do not require a biological dataset, broad autotuning, exhaustive validation or a speedup. Stronger production/scientific claims can be qualified later without blocking experimental implementation now.
