# Development and Project Control

## Current build support

The active target is the portable CPU reference library. The documented and
supported standalone toolchain is Clang 18 with matching `clang-scan-deps`,
CMake 3.28 or newer, and Ninja. CMake rejects non-Clang compilers and
non-Ninja generators for this C++20 module build. This is an independent
component: it does not configure the Cellerator parent build, Python, Torch,
or CUDA. Velocity itself adds no HDF5 or Zarr codec target; the installed
AnnData++ package controls whether its array/dataset interfaces require those
storage libraries.

Install AnnData++ and the CPU-only `CelleratorBinary16` package separately,
then configure from this directory:

```sh
cmake -S . -B build/standalone -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install;/path/to/binary16/install" \
  -DVELOCITY_BUILD_TESTS=OFF \
  -DVELOCITY_BUILD_EXAMPLES=OFF
cmake --build build/standalone
```

AnnData++ supplies the independent core array, axis, and dataset interfaces.
The small `Cellerator::binary16` helper converts FP16 values and does not
require the full Cellerator runtime. The installed target is
`velocity::velocity`; its public C++20 modules are `velocity` and
`velocity.precision`. The active component exports no CUDA or ISA backend.

The CMake project also contains opt-in native tests, examples, and historical
qualification drivers. Those targets are not qualification evidence by
being present. See [validation](validation.md) for the distinction between
old receipts and the current caller-scheduled reference API.

## Reproduce current native checks and source oracles

The current source receipt is [thin-reference.json](../tests/results/thin-reference.json).
It records the source identity, native test results, numerical comparisons,
and strict source-diagnostic exception. The commands below use fresh build
directories and write oracle outputs under `/tmp`, leaving historical result
files untouched. Use installed AnnData++ and `CelleratorBinary16` prefixes.

```sh
deps="/path/to/anndatapp/install;/path/to/binary16/install"
cmake -S . -B build/thin-reference-release -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="$deps" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVELOCITY_BUILD_TESTS=ON \
  -DVELOCITY_BUILD_EXAMPLES=ON \
  -DVELOCITY_BUILD_QUALIFICATION=ON
cmake --build build/thin-reference-release
ctest --test-dir build/thin-reference-release --output-on-failure
```

For the recorded AddressSanitizer/UndefinedBehaviorSanitizer configuration,
configure a separate build and set `detect_leaks=0` as recorded; this does not
run LeakSanitizer:

```sh
cmake -S . -B build/thin-reference-asan -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="$deps" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVELOCITY_BUILD_TESTS=ON \
  -DVELOCITY_BUILD_EXAMPLES=ON \
  -DVELOCITY_BUILD_QUALIFICATION=ON \
  -DVELOCITY_ENABLE_SANITIZERS=ON
cmake --build build/thin-reference-asan
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build/thin-reference-asan --output-on-failure
```

Run the deterministic and stochastic source oracles only with the pinned
scVelo checkout and reference Python environment. Both tools accept explicit
report and artifact destinations so they do not replace historical milestone
JSON:

```sh
reference_python=/path/to/reference-env/bin/python
export SCVELO_SOURCE=/path/to/scVelo
"$reference_python" tools/qualify.py \
  --driver build/thin-reference-release/velocity_interop_driver \
  --output /tmp/velocity-thin-deterministic.json \
  --artifacts-dir /tmp/velocity-thin-deterministic-artifacts
"$reference_python" tools/qualify_stochastic.py \
  --driver build/thin-reference-release/velocity_stochastic_interop_driver \
  --output /tmp/velocity-thin-stochastic.json \
  --artifacts-dir /tmp/velocity-thin-stochastic-artifacts
```

The stochastic source report preserves the strict initial-offset diagnostic
exception and can exit 1 even when all estimator gates pass. Check its
individual gates and recorded exception rather than treating the process exit
as a tolerance pass.

The TSan worker-thread check and allocation instrumentation are separate
receipts: the TSan build disables the replacement `new`/`delete` hooks to
avoid a runtime symbol conflict, so allocation evidence comes from the native
and ASan/UBSan runs. The detailed conditions and limitations are in the
receipt.

## Historical M1/M2 build and qualification workflows

The commands and receipts below preserve the earlier serial/scalar and
accelerator-era workflow. They refer to older source snapshots, not to the
current thin caller-scheduled API. The older accelerator sources and comparison
runners are preserved under the parent checkout's quarantine; current CMake
has no accelerator option, toolkit discovery, ISA target, or selectable
backend. Do not use these commands or receipts as a current qualification
claim.

## M1/M2 historical numerical and memory qualification

Milestone 1's optional Python tools use `tests/requirements-reference.txt`, a reference
environment, and the pinned scVelo checkout selected by `SCVELO_SOURCE`.
The source-comparison route executes verified original production definitions
extracted from the source AST, avoiding unrelated Scanpy import-time machinery.
Milestone 2 also runs an independent masked-fit oracle because the chosen
consistent row-mask behavior deliberately differs from scVelo. The tools do
not download biological data.

```sh
reference_python=/path/to/reference-env/bin/python
export SCVELO_SOURCE=/path/to/scVelo
"$reference_python" tools/qualify.py --driver build/release/velocity_interop_driver \
  --output /tmp/velocity-m1-qualification.json \
  --artifacts-dir /tmp/velocity-m1-artifacts
"$reference_python" tools/benchmark_reference.py generate /tmp/velocity-m1-canonical.bin
python tools/measure_build.py --output /tmp/velocity-m1-module-build.json
"$reference_python" tools/qualify_stochastic.py \
  --driver build/release/velocity_stochastic_interop_driver \
  --output /tmp/velocity-m2-qualification.json \
  --artifacts-dir /tmp/velocity-m2-artifacts
```

The historical paired CPU benchmark runners write fixed
`benchmarks/results/milestone-1.json` or `milestone-2.json` paths and matching
artifact names. They are intentionally omitted from runnable commands here;
their preserved reports are the historical evidence. A reproduction belongs
in a disposable source copy with separately reviewed result paths.

Stochastic numerical qualification compares the pinned scVelo source and the
independent consistent-mask oracle on both supplied fixtures, including gene
subsets, intercept combinations and trimming modes. Its compact result is
`tests/results/milestone-2.json`. The paired benchmark runner generates a fixed-seed 4,096-cell, 2,048-gene
input when `--input` is omitted, then records results in
`benchmarks/results/milestone-2.json`; `--scaling-only` adds bounded workspace
scaling controls. Choose an allowed idle CPU and set it with `--cpu`. The Milestone 2 module-build output path keeps the historical
Milestone 1 result intact.

The archived Milestone 2 stochastic qualification command exited 1 because
four FP32 source-initialization offsets exceeded their original tolerance. Its
report separately records passing estimator, independent FP64 initialization,
and production-class gates. See the explicit [qualification exception](validation.md);
the accepted precision-related exception did not turn that strict source
diagnostic into a tolerance pass.

Do not interpret the Milestone 1 records as stochastic-fit evidence.

Select an allowed idle CPU for `--cpu`; workers run sequentially with one
BLAS/OMP thread, a 30-second timeout and a 2 GiB address-space cap. Reports
record source, binary and input hashes. Compact results live under
`tests/results` and `benchmarks/results`; generated fixtures, raw runs, logs,
and disposable source/build copies remain ignored. The build measurement
changes only its disposable copy and checks private edits preserve public
BMIs and consumer objects. See [validation](validation.md) for evidence limits.

## Contribution and scope

Read [AGENTS.md](../AGENTS.md), [STYLE.md](../STYLE.md),
[architecture.md](architecture.md), and [validation.md](validation.md).
Follow the written, protected
[Cellerator coding guide](/home/tumlinson/Cellerator/coding_style.md).
Keep numerical calculations compact and memory/ownership operations explicit.

Inspect current source, method options, and attribution before adapting scVelo
or accelerated-ports code. Preserve upstream license notices for any derived
implementation. Documentation of an opportunity is not a measured improvement.

Use configured Codex subagents for useful bounded work. Coordinate overlapping
files and shared compute. Preserve other agents' changes, upstream checkouts,
supplied data, native workflow state, and historical evidence.

## Project Control

Active work on this relocated component uses the parent Cellerator checkout's
Project Control registration and Todo authority. The `velocity-cppm` workspace
identity and setup record below are historical. The original
`.todo-orchestrator/` now resides at `/home/tumlinson/velocity`; relocation did
not move or recreate that workflow state.

### Historical velocity-cppm identities and workflow

The former workspace and repository alias were both `velocity-cppm`; its
intended authority ID was `velocity-cppm-authority`. Registration used the
actual root `/home/tumlinson/velocity.cppm`. Its local origin was
`git@github.com:tumlinso/velocity.cppm.git`.

The installed [Todo Orchestrator skill](/home/tumlinson/.agents/skills/todo-orchestrator/SKILL.md)
describes supported workflow routes. Routine work in the registered workspace
uses `next_task`, then the opaque handle with `inspect_task`, `coordinate_task`,
and `finish_task`. Observer reads require no claim. `next_task` is not a
readiness check because it can claim or resume work.

Canonical state for that former workspace remains in
`/home/tumlinson/velocity/.todo-orchestrator/authority.sqlite3`, with
project identity in its `project.json`. Those files and SQLite sidecars remain
private/ignored in the old checkout. Never hand-edit the database, generated
projections, leases, or recovery state. Keep owner and profile credentials
outside the checkout; do not publish tokens, private previews, or runtime
artifacts.

Authored planning files are reviewable inputs. Todo's accepted plan and task
state are authority. [Milestone 1](../planning/milestone-1.json) was accepted
through supported authoring. Context assembly failed after native attempt
admission; the user explicitly instructed work to proceed without a handle.
The [workflow receipt](../planning/milestone-1-workflow.json) preserves this
distinction. Local results are not a native accepted outcome.

### Historical setup verification

[The former setup receipt](../planning/setup-receipt.json) records the result,
installed versions, identities, and limitations for `velocity-cppm`; it does
not describe the active Cellerator authority. Check those historical details
independently if they are needed:

1. The registry maps the intended alias to the intended repository root.
2. The initialized Todo authority has matching project/repository/authority
   identities and the expected current plan/task state.
3. Native integration refs and private provider runtime directories exist.
4. Host credentials are scoped to the exact project, authority, principal,
   profile, and permitted actions; routine profiles do not receive the owner
   bearer.
5. A fresh Project Control process can retrieve exact source and semantic
   workflow state. An existing connected process may retain an older registry
   or provider result; report that separately.

Service health alone is not project readiness. A partial response caused only
by an unavailable optional context extension differs from missing authority,
credentials, or source access. Preserve active attempts and unrelated sessions
when resolving setup problems. Do not restart shared services or change their
permissions merely to silence a diagnostic.

## Artifacts and checks

Keep generated biological data, builds, environments, binaries, profiler output,
and raw measurement campaigns ignored and preserved locally. Commit source,
small reproducible fixtures/generators, compact evidence, and source identities
when publication is authorized. Commit and push only with user authorization.

For documentation changes, check relative links, source-reference provenance,
JSON syntax, status claims, secret exclusion, and `git diff --check`.
Run affected native, installed-consumer, sanitizer, numerical, and measurement
checks as appropriate. Do not report unexecuted checks as passing or repeat a
measurement campaign without a new evidence gap.
