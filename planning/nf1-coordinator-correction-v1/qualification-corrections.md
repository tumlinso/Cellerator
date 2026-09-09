# Native qualification corrections, 2026-09-09

P01 received narrowly scoped ownership of `include/Cellerator/execution/launch_bindings.hh` and `docs/current_implementation.qmd` at revision 7155. Task brief v2 was published at 7156. This admits real host bindings through the existing semantic owner without inventing GPU metadata. Actual-source qualification completed P01 at 7160, source `0523ba4ec6af3fa2553d3386b602a8f4ab238aae`.

V01 gate evidence `b7d272e4-4de6-4732-879e-5dbedf406a71` failed before launching a device test: the controller observed two idle samples in 14.6356 seconds but needed three inside its default 10-second observation window. No competing process or GPU utilization was observed. The original failed receipt remains historical evidence.

Skills commit `ba99642a79a795967e0ba6d94cb62aa43eaebf90` adds validated forwarding of the controller's existing quiescence options. Twenty-two focused tests passed. The installed pinned candidate `/home/tumlinson/project-control/.venv-nf1-quiescence-ba99642` was tested on an isolated port before atomic wrapper promotion; the previous `.venv` remains intact for rollback. Live health, readiness, and version returned HTTP 200 after restart. Release manifest SHA256: `9c1dba4d8b46102932ec76c1e8a1583bdd922a270696025b80783192128284df`.

The V01 gate amendment at revision 7162 allows 60 seconds and retains the required three consecutive idle samples. It does not certify CUDA execution. The revised source must still pass the actual native gate, including the shared evidence lock, live lease, and survivor lifetime regression. Existing persistent Codex stdio processes require a separately verified runtime transition; changing the launcher alone does not reload them.
