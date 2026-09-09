# Coordinator lease recovery and renewal, 2026-09-09

The first-class resume path updated claim heartbeat without extending expiry. Long-running coordinator claims therefore expired despite resumption. The existing normal claim pulse owner now renews claim and associated lease expiry during authenticated active-session resume (`bc28d971e5ed76ac04f8fbc3afdbc2f84c43254d`). Foreign or already expired claims cannot be renewed this way.

Canonical owner recovery now handles only an expired clean read-shared coordinator seat, retaining ordinary refusal for writable/dirty work or live mutable operations (`77d0cda8caa45fae6fbf865700dce0543a276e4e`). A selected unswept expired claim can be expired through the existing claims owner in the same guarded recovery transaction (`a0717069fb5a908b68137896c5e920d5a53e0bcc`), without sweeping unrelated claims. Authority revision and safety state are revalidated under the transaction lock. Task, lane and capability changes are audited; source and historical claims are preserved.

Validation: 42 recovery/regression tests; 44 renewal/regression tests; 51 tests after the unswept-claim follow-up. The final pinned candidate is `/home/tumlinson/project-control/.venv-nf1-coordinator-a071706`, release-manifest SHA256 `a3b7f9fbda17cb1b0bab1ac17b00d5ee8358db7782931702c9e6441869ae6c92`. Source, frozen skill and installed module hashes match. Its isolated health/readiness/version endpoints returned HTTP 200 before promotion. Previous runtime candidates remain intact.

Applied through `project-control admin recover`, after concrete task-specific previews and canonical TTY confirmation:

- CE-NF1-COORD: recovery revision 7195, audit `80cc11fd-f822-4826-85e0-f57ad168270f`; reacquired 7196 and resumed 7198. Live claim heartbeat `2026-09-09T13:14:35Z` and expiry `2026-09-09T15:14:35Z` prove actual renewal.
- GH-NF1-COORD: recovery revision 137, audit `b77e36a9-f87d-4bc2-aac3-5ab171f17108`; reacquired 138. Its expired claim had not been swept because the idle next-task transaction rolled back. Recovery expired only that selected claim and requeued the same coordinator seat.

Persistent old MCP processes are not silently rewritten. A fresh process in the pinned candidate invokes the same public capability-enforcing WorkflowProtocol for coordinator resume and CUDA qualification using opaque workflow handles. No raw claim credentials, direct authority writes, local models, or subordinate local-worker execution are involved. Canonical task scopes, lanes, dependencies and gates still apply. Subsequent heartbeats must use corrected `next_task`, not sync alone.
