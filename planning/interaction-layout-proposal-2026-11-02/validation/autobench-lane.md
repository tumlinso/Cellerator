# Factorized interaction validation lane

## Active source and registration

- Isolated workspace: `planning/interaction-layout-proposal-2026-11-02/measurement`
- Registered benchmark: `cellerator-factorized-interactions`, revision 12
- Registration: `bench/factorized_interactions/registration.toml`
- Immutable source: `.autobench/data/factorized-interactions/source-v14`
- Source closure SHA256: `3c402181eea792558d46d1ac9abde5dd6a7ec0f477786694d7d33241621eb007`
- Source manifest SHA256: `8a9fc6f09d7437be8347ec877dbd07ff765d414a1ad4e50ef0ccc5fbd29080c1`
- Source size: 4,819 files / 21,431,444 bytes
- Archive: `.autobench/data/factorized-interactions/cellerator-factorized-interactions-source-v14.tar`
- Archive SHA256: `320f17ada8402b620982a7c2ea66fd40a2339504604eb0b7f7ce19497a6cf020` (26,009,600 bytes)

The closure contains the native core, independent oracle, test corpus, candidate
and control sources, build helpers, `CONTRACT.md`, `PERFORMANCE_PLAN.md`, and a
copied read-only Baseplane dependency. It contains no symlinks. Candidate source
is compiled into a separately manifested native runner by the build helper;
it does not alter the baseline runner. Generated Python bytecode is excluded.
No Baseplane files were added to the Cellerator source tree.

## CPU checks and configure rehearsal

The local independent oracle/wire/workload unit suite passed 32/32 tests in
6.84 seconds with bytecode generation disabled. Seven changed Python files
parsed with `ast.parse`; the sanitizer suite `--help` command parsed and showed
the registered arguments. These local checks are not substitutes for the
registered source-bound host packet.

The exact v14 archive and staged helper passed a baseline configure-only
rehearsal at `/tmp/factorized-interaction-v14-configure`; the extracted source
closure matched the archive SHA and CMake generated successfully with CUDA
12.9, SM70, Python 3.13, pybind11, Torch discovery, and staged Baseplane. The
candidate-only configure helper then passed at
`/tmp/factorized-interaction-v14-candidate-configure`. Neither rehearsal built
targets or launched a GPU. CMake reported missing NVTX3/NVRTC short-hash
warnings and a nonfatal `git` metadata probe because the immutable extraction
has no `.git` directory.

## v14 build and host-oracle failure

Autobench accepted revision 12 with the host's 600-second command ceiling. The
source-bound build succeeded as Autobench attempt
`e7c89cfa-1f5f-41aa-b20a-fd019f4c605d`, Scheduler job
`f2b294b8-0549-4be4-9933-916b41701e27`, build fingerprint
`4c5ab6f5233dc5d24b563c35349a132d8248d6531c4aba68b82309804b91c199`. It produced
the baseline runner, C++ training example, host-test
binaries, Torch extension, and candidate runner. The verified Autobench build
output identity is `9a880da6909ef0cf57bd4627aaa08a4d507c4cfcfc7b2d4e7cf1169a33c77e16`;
build-manifest SHA256 is
`3640c3c3e95bc29f7de1b71a34a85c0f9b3acf2f971be5c04f8eb99a103c5153`. The
manifest reports the same v14 source closure. Runner SHA256 is
`f697d120ba70fe88527ab72dbd9ec8b2a1013786cee979742825c10be936ab5d`, training
binary SHA256 is `14dec7198dd35f911065b6b0c3f668e6bba27daf6843b3a6156c81af6b3c30c7`,
and Python extension SHA256 is
`9a7b1e9f2a645b428b0c24bd6faa98408687cade6fcb8bae5233414613178a0f`. The
candidate build-manifest SHA256 is
`534742e925eb88c9a5ae7fc36e55888ba6b3135e4c1808079d8d16b2362c2c7d`; candidate
runner SHA256 is `1fdd3548da0e8cfb3e285c8ed2538ca5694c5404f0bd752954c3ffadafc39b8b`.

The dependent `host-oracle` packet was submitted as attempt
`17518882-014f-42d1-9f2f-2b9ba95cee11`, Scheduler job
`cf60fa56-0f2f-4f20-be93-90f560562da0`, and failed with cleanup verified. It
ran all 32 Python tests: 31 passed, and
`test_wrapper_serializes_invokes_and_checks_oracle_wire_stub` failed when its
stub child imported NumPy. The captured Python 3.12 venv executable was
canonicalized by `validate_runner.py` using `Path(command[0]).resolve()`, which
followed the venv symlink to `/usr/bin/python3.12` and dropped venv packages.
An independent run of the full frozen-source Python suite under a 64 MiB
`RLIMIT_FSIZE`, bytecode disabled, and output-local temporary directory
reproduced 31/32 in 7.60 seconds. Failure artifacts are retained at
`validation/host-oracle-v14-failure.stderr.log`,
`validation/host-oracle-v14-failure-attempt.json`, and
`validation/host-oracle-v14-repro.{json,stdout.log,stderr.log}`. The fix is
owned by the wire/validator contributor; no tolerance or oracle changes are
planned. No GPU job has been submitted.

The managed host packet launches three manifest-hash-verified C++ test
binaries directly and writes logs only under its `{output}` directory, avoiding
CTest writes in `{build}`. `autobench explain` still reports the same build
output identity after the failed host packet.

The registered GPU dependency is baseline full-VJP correctness (including the
200-step C++ training example and prepared Torch lifecycle), then serial
Compute Sanitizer coverage for packet/direct baseline routes plus selected
`direct_unique_output` and `minimal_direct` cases. The sanitizer suite derives
the candidate runner and manifest from the immutable parent build manifest and
checks actual candidate/backend/layout metadata; memcheck requests full leak
checking. The timed KDA candidate comparison is intentionally separate and is
not registered in this first native qualification revision.

## Preserved prior attempts and identity issue

Revision 11's CPU build succeeded as Scheduler job
`82180fe4-bc50-4d5a-8dd9-b6b7f2af7742`, Autobench attempt
`6764e7a8-ab30-4210-93c8-5dd238f9992a`. Its host packet succeeded as job
`637e7ccb-591b-4d5a-b6be-fb154e1ef707`, attempt
`f05505cc-e0f9-4b43-b992-fce9d9cd6f74`: 28 Python tests and 3/3 host binaries
passed. The host packet ran CTest, which wrote `cmake/Testing/Temporary` logs
inside the immutable build tree and changed Autobench's build output identity;
that evidence is therefore not used for v14. Revision 11's duplicate build
attempt `d32f2cff-f030-4351-9e03-21ce6561edb4` / Scheduler job
`2a3beca7-6787-4f10-be54-d31be3262b93` was reconciled as failed with
`FileExistsError: build-local source extraction already exists`. Its receipt
and output remain preserved. Revision 12 replaces that path with direct,
manifest-hash-verified host binary launches from `{output}` and no CTest.

Earlier build failures are also retained in the Autobench result store. No
device correctness, sanitizer, ATen runtime, or performance result is claimed
until its registered packet completes with matching source and binary identity.
