# Autobench build-identity diagnosis, 2026-11-02

The isolated `cellerator-factorized-interactions` revision 11 build completed
under Scheduler job `82180fe4-bc50-4d5a-8dd9-b6b7f2af7742` and Autobench
attempt `6764e7a8-ab30-4210-93c8-5dd238f9992a`. Its build fingerprint is
`4fe29da9c8f0687a592480c155ee145dbf315b798ce83e2057164dbe67c9fc71`.
The dependent host-oracle attempt `f05505cc-e0f9-4b43-b992-fce9d9cd6f74`
passed 28 Python tests and three CTest cases, then the next reconciliation
reported the build as stale and submitted a second build attempt.

Autobench's `_build_output_identity` hashes every regular file under
`{build}`, except `SUCCESS.json` and paths beneath `__pycache__`. The host
packet invoked `ctest --test-dir {build}/cmake`, which wrote these files under
the already recorded build directory:

```text
cmake/Testing/Temporary/LastTest.log
cmake/Testing/Temporary/CTestCostData.txt
```

A read-only reproduction of the engine's digest gives:

| Hashed files | SHA256 |
| --- | --- |
| Original identity recorded in `SUCCESS.json` | `07ced25000a59205a1f429743357355286455263bb48b3b6c30e65629a91b8fc` |
| All current files | `0a8874d22b1d6e934eb502134f8a3c1ace85af151a1ad1f348885f15dbfa334d` |
| Current files excluding only the two CTest temporaries | `07ced25000a59205a1f429743357355286455263bb48b3b6c30e65629a91b8fc` |

The last row reproduces the recorded identity exactly. It is diagnostic only:
no managed build, success marker, database, or receipt was edited or removed.
Autobench correctly rejects a build whose recorded content has changed.

The generated `CTestTestfile.cmake` shows that the three cases execute the
preparation, pattern-binding and CUDA host-contract test binaries directly,
without command arguments or a special environment. A supported repair is to
verify those three executable hashes against the build manifest, run them with
60-second individual limits and working directories under `{output}`, and
retain their individual results and logs there. This preserves the underlying
checks while keeping the build tree immutable. It requires a new registered
source/build identity; it does not justify deleting the two existing files or
altering `SUCCESS.json`. The published earlier CTest result remains historical
evidence for its original attempt.

Two additional source-audit findings were sent to the measurement owner before
device submission:

1. The million-mechanism case serializes 72-byte records into a single file of
   at least 72,000,000 bytes. Autobench's worker applies a 64 MiB per-file
   `RLIMIT_FSIZE`; this case therefore needs bounded shards or another
   supported representation of the same complete wire bytes. The frozen case
   and independent oracle must remain unchanged.
2. The optional prepared-Torch runtime fixture numbers `logical_id` from zero,
   but native preparation rejects zero. Logical IDs must start at one; tensor
   operand and target ordinals remain zero-based.

The native correctness, sanitizer and timing drivers place their explicit case,
result, metadata, metrics and report files under the supplied artifact/output
directories. Their Python entrypoints disable bytecode writes; Autobench also
excludes `__pycache__` from the build digest and places temporary/cache variables
under per-attempt scratch. This is a source audit, not a proof that every
third-party library is side-effect-free; the first completed device packet must
still leave the build identity current.

## Host-packet repair check

The revised `run_cpu_suite.py` with SHA256
`96c75b51e54a4f75337b5e2eb6aaa3a6d47aed18714cdc7bfb15eeda477aded7`
was exercised against a temporary synthetic build directory containing copies of
the previous manifest and its three manifest-verified host executables. Its
source argument pointed to the current live checkout. The run passed 32 Python
tests and all three host executables. The digest of all files in that temporary
build was
`9e3c185bf0bf2cf17f1a91dd8cc26ba17de3d56b3a787731593e4cafdc2f22f5`
both before and after the run. The script bytes also remained unchanged.

This is an independent CPU check that the repaired harness does not mutate its
supplied build directory; it is not a source-bound Autobench run and does not
qualify CUDA. The managed build directory was not changed during this check.

## Bounded wire repair

The transport now preserves `CFINT001` and writes its ordered stream as at most
32 MiB per shard. The validator verifies the SHA256 of the complete stream
before and after invoking the runner, which accepts a first `--input` followed
by ordered `--input-shard` paths and concatenates the original bytes before
parsing. Legacy one-file cases use the same `--input` spelling as before.

The independent CPU test enforced a real 64 MiB `RLIMIT_FSIZE`, generated the
full fixed million-mechanism case, and read its three files through the same
host reader as the native CUDA runner. Shard sizes were 33,554,432,
33,554,432, and 4,891,248 bytes. The 72,000,112 reconstructed bytes matched a
separately assembled canonical fixture SHA256. Other checks compared a small
cross-boundary request byte-for-byte against the original unsharded encoding,
and rejected reordered, corrupted and missing shards. The complete Python
reference/wire/corpus suite passed 32/32, and the modified native runner compiled
and linked for SM70. These checks did not execute CUDA.

## Worker interpreter identity

The next captured source-bound host packet, attempt
`17518882-014f-42d1-9f2f-2b9ba95cee11`, failed one smoke test while the other
31 tests passed. Its retained failure report is
[`host-oracle-v14-failure.stderr.log`](host-oracle-v14-failure.stderr.log).
The failure was independently reproduced against the same immutable source
tree using Autobench's Python 3.12 virtual-environment interpreter. The
validator converted its runner executable path with `Path.resolve()`, which
followed the environment's Python symlink to `/usr/bin/python3.12`. The child
then lacked the environment's NumPy package.

The validator now uses an absolute path without resolving symlinks. Its SHA256
is `5f645de32c0381de68ce31eb8e37e0f6e19a0a80c279f1a22c75aa1424cbf4cc`.
The final regression test file has SHA256
`394b575b1a3451299f0adce1f982cae1a80c1f1079a1089e0ee59871a1b9b244`.
The live suite passed 33/33 tests under the exact Autobench interpreter before a
diagnostic-only improvement to the new test's error message; all three final
smoke tests then passed under that interpreter. A temporary negative control
restoring the old `resolve()` expression made the new regression fail with its
expected interpreter-symlink message. The managed build and state were not
modified during these CPU checks.
