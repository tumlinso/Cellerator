# Independent review: future process observation candidate

**Assessment:** No confirmed defect in the scoped process-instance attribution
was found in the fixed candidate below. This is a CPU/source review, not a GPU
qualification or permission to submit a job. The original failed packet
`45bab455-239f-4e32-8ebd-fd33ae03e979` remains rejected.

Reviewed working-tree SHA-256 identities:

* `bench/chromatin_geometry/scripts/run_packet.py`:
  `8cf4187cd316030468e04947bb1182059ab80abf4e2251620d7e7035c1e38825`.
* `bench/chromatin_geometry/scripts/kda_driver.py`:
  `220e2b2340cead9b4ff857d361e02026604de0e297d1ca58d78837cf1afa8e7c`.
* `bench/chromatin_geometry/scripts/kda_evaluate.py`:
  `ef0b67da71926fe2bb603c8c248019f6092c09591306744d23e3f66c35577345`.

## Identity and rejection

In `kda_driver.py:52`, the native launch and its start event occur under the
same packet-local exclusive flock used by `run_packet.py:60` for the process
query, active-record lookup, `/proc` start ticks, and ancestry check. The driver
uses `waitid(WNOWAIT)` before reacquiring the lock for the actual `wait()` and
end event. With the tracked driver as the only reaper, an exited child remains
allocated as a zombie while the query holds the lock, so its PID cannot be
reused during the observation. The monitor separately checks matching PID,
device, invocation, start ticks, successful output hash and
`spawn_upper <= process_query_start <= process_query_end <= reap_lower`
(`run_packet.py:175–216`). A foreign device, unknown PID, bad start ticks,
missing or invalid lifetime data, unreadable identity, or failed process query
prevents a clean result. Utilization and process queries remain sequential,
and the existing one-second retrospective utilization rule and sampled
observation limits remain distinct from the process-instance proof.

I ran `test_process_observation.py` (3 tests), including the real child that
exits while a delayed mocked process query holds the lock, and
`test_serial_telemetry.py` (6 tests); all nine passed. The hardware process
query itself is mocked. The preexisting `test_kda_driver.py` also passed.

## Failure paths and provenance

The driver's spawn/start-event exception path kills and reaps the child while
the lock is held; the `waitid` exception path also locks before kill/reap.
An independent forced start-event failure left no child in `/proc`. Separate
CPU execution produced two million characters on each of stdout and stderr
without a pipe deadlock; both streams were returned, with paired events. The
monitor and tracked driver use temporary files instead of undrained pipes.
The reviewer did not exhaustively inject process termination or filesystem
failures. In particular, an unexpected monitor termination is not itself
proof of cleanup: the actual Scheduler-owned unit and its verified cleanup
remain required.

`kda_setup.py:49–61` binds the driver from the extracted closure, and the
evaluator imports the adjacent closure driver for supplemental rounds. The
historical archive `f131948000849270e22c392459264fa599566f4a1d51492a09e72d9e1d8f72c7`
contains the prior driver, not this candidate. Therefore these working-tree
hashes alone do not establish what a future packet executes. Capture the
candidate into a newly identified closure and bind its matching build receipt,
binary and packet snapshot before relying on it; `run_packet.py:308–313`
rejects a frozen build receipt associated with a different archive. The
source review does not establish the other study measurement gates, host-wide
exclusivity or absence of activity between telemetry samples.
