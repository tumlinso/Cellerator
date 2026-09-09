# NF1 coordinator queue correction

The accepted initial plan queued epic `CE-NF1-0000` as the only coordinator task. The installed claim engine excludes an epic until its children complete, preventing acquisition of the coordinator seat needed to execute them.

This schema-3 maintenance delta adds `CE-NF1-COORD` as a normal child task and places it before the unchanged epic in `CE-NF1-L-COORD`. The existing run, charter, root, lane role, workspace mode, implementation queues, gates, interfaces and cross-project closure order are preserved. No source capability is marked complete. The coordinator finishes after local M90 and before epic closure.

Apply only through supported transactional plan administration following a fresh validation/diff. The user explicitly authorized this correction and related routine repairs on 2026-09-09. This is an additive correction to the sealed original delivery, not a replacement task database.

The coordinator task reserves a project-local named seat lock, satisfying parallel-safe scope admission without claiming production paths. Reapplying a queue correction while a lane is active is not a recovery procedure.
