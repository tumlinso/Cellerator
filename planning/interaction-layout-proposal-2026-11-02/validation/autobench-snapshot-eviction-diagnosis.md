# Autobench in-flight snapshot eviction, 2026-11-02

The v15 build attempt `2cf1f367-5b24-449e-939b-6d46340648d2` / Scheduler
job `9172dae5-892f-47da-a9e7-ace95c12e419` failed before starting its command:
its working directory, the captured source snapshot, no longer existed. The
[Scheduler receipt](v15-build-failure.scheduler.json) records terminal failure
and verified cleanup. No GPU was allocated.

The installed Autobench engine and the separately checked source checkout have
the same SHA256:
`b8cb4768caaf0935d819f3f4b05fd7026f9817f7d1360e7d6241af7f9440f6a8`.
Its `_submit` method performs quota-driven `_evict_safe` **before** creating the
new attempt record. At that point the just-captured snapshot is not among the
paths pinned by unresolved attempts. If the 128 MiB scratch reservation would
exceed the workspace cap, eviction can remove that snapshot and then submit a
job naming the removed directory as its cwd.

The measured workspace state is consistent with this path: before submission,
the measurement owner observed approximately 963 MB retained against a 1 GiB
cap; after failure the managed tree occupied approximately 751 MB and the
snapshot directory was empty. The engine's eviction target under that pressure
is 80% of `(1 GiB - 128 MiB)`, or 751,619,276 bytes. This observation is
circumstantial for the historical attempt; no engine instrumentation or
authority-state edits were used.

## Isolated reproduction and proposed correction

[`autobench-snapshot-eviction-repro.py`](autobench-snapshot-eviction-repro.py)
uses the actual installed engine with a fresh temporary directory and an
in-memory store stub, never the campaign's authority or a Scheduler job. With
800 bytes of retained data, a 1,024-byte cap and a 256-byte reservation, the
installed engine deleted the newly captured 200-byte snapshot, reached 600
bytes and would permit submission. A candidate engine that pins the in-flight
snapshot and build kept the snapshot, remained at 800 bytes and would refuse the
reservation rather than submit a missing cwd.

[`autobench-snapshot-pin.patch`](autobench-snapshot-pin.patch) is a minimal
proposed correction for review: protect the currently submitted snapshot/build
during eviction and reject a missing snapshot before Scheduler submission. It
was applied and tested only to a temporary copy of the installed Python
package; neither the Autobench checkout nor its installed release was changed.
The reproduction does not replace Autobench's own regression suite or prove
all concurrent eviction paths.

The installed-engine reproduction and its temporary patched-copy check can be
repeated with:

```sh
/home/tumlinson/.local/state/autobench/releases/0.1.0-4f4ee30-py312/venv/bin/python3 \
  planning/interaction-layout-proposal-2026-11-02/validation/autobench-snapshot-eviction-repro.py \
  --verify-patch
```

## Supported workspace workaround

Autobench's public `init` command accepts `--retained-cap-gib`; reinitializing
this already initialized **isolated measurement workspace** with a 2 GiB cap,
`--no-activate`, and `--agent-instructions skip` updates its workspace
retention setting through the supported interface. Omitting Scheduler URL and
token arguments preserves their existing values. The installed host retention
ceiling is 4 GiB. The workspace cap governs retained disk space; it does not
alter Scheduler GPU/CPU admission or the host hard ceiling. After the failed
attempt's cleanup is reconciled, ordinary capture can materialize the same
unchanged source again. Existing receipts and unrelated paused workspaces do not
need manual modification or deletion.
This creates headroom for the observed workspace; it does not repair the engine
or guarantee that a future build cannot reach the new retention limit.
