# Environment quick fixes

Keep this list short: recurring issue, verified fix, date. Recheck paths before use.

- **2026-09-06 — CUDA wrappers select missing 13.1 tools.** Use CUDA **12.9**:
  `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc` and
  `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer`.
  Set `CUDACXX` to that nvcc and put those two directories first on `PATH`.
- **2026-09-06 — Shared Project Control MCP session can resume another lane.**
  Pass the assigned lane's explicit current queue-head `task_id` to `next_task`,
  verify the returned lane, and retain its own workflow handle.
- **2026-09-06 — New evidence under `docs/` is ignored by Git.** Use
  `git add -f` for the exact authorized evidence paths and inspect the staged diff.
- **2026-09-06 — ctxpp reports stale metadata/missing compile database and
  intrinsic parse errors.** Use the skill's readable canonical-source fallback
  for bounded inspection; do not treat the stale index as semantic authority.
