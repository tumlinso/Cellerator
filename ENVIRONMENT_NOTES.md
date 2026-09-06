# Environment quick fixes

Current operational guidance, checked 2026-09-06. Historical incident details
remain in `docs/semantic_spine_v1/recovery_log.md`.

- **CUDA toolkit:** local wrappers can select missing 13.1 tools. The verified
  toolkit is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9`. Controller specs
  accept `toolchain: {root: <path>, require_sanitizer: true}` and resolve compiler
  and sanitizer together. An invalid explicit selection fails closed.
- **Build before execution:** foreground controller builds use one structured
  `benchmark.build_argv`, before GPU reservation. Top-level `build_argv` is now
  rejected. Declare `binary_paths` to record the tested files' SHA-256 values.
- **GPU workflow gates:** new command gates should declare `cuda` resources and
  toolkit settings. Explicit gate calls and completion reruns then acquire the
  controller reservation/mutex themselves. The SS1 log's temporary outer wrapper
  describes historical recovery, not the new gate contract.
- **Lane selection:** pass the assigned queue-head `task_id` to `next_task`,
  verify its returned lane, and retain that lane's workflow handle. One session
  cannot claim a different active lane. Completion records the producer worktree
  commit separately from authority/main; declared artifacts must exist there.
- **Runtime updates:** build a candidate with Project Control's installer. New
  releases contain a frozen Skills snapshot and digest-pinned release manifest;
  development source edits do not invalidate a deployed release. Promote HTTP
  and stdio together. Existing MCP clients require a fresh connection.
- **Git:** Semantic Spine evidence and `tests/semantic_spine/core` no longer need
  force-add. Crash-dump patterns are root-scoped. Other historical documentation
  directories retain explicit ignore policy; inspect exact staged paths.
- **ctxpp:** `.ctxpp.toml` selects `build-ss1/compile_commands.json`. Refresh it with
  `cmake -S . -B build-ss1 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, then run the skill's
  `init`, `doctor`, and `status`. Retrieval refreshes stale translation units;
  missing compile commands or parse failures prohibit semantic rewrite claims.
  The GCC-private-header injection causing the earlier intrinsic errors is fixed
  in both translation paths; the descriptor-test semantic probe now parses cleanly.
  Use readable canonical-source inspection when other parsing remains unavailable.

### RU1 worktree builds and sanitizer selection (2026-09-06)

Managed worktrees are nested below `.git`, so sibling discovery does not find
Baseplane. Configure with `-DBASEPLANE_SOURCE_DIR=/home/tumlinson/Baseplane`;
build output remains in the Cellerator worktree. CUDA 12.9 sm70 builds use
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc` with
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12` and `-DCMAKE_CUDA_ARCHITECTURES=70`.

A controller sanitizer recipe resolved an absent CUDA 13.1 tool while returning
an outer success status. Invoke the verified executable
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer`
explicitly, set a nonzero `--error-exitcode`, and require the raw sanitizer
summary as well as the actual test output. The corrected readiness and native
runs reported zero memcheck errors. Controller quiescence uses a 60-second
budget to fit all three required idle samples; retain the samples and mutex.
