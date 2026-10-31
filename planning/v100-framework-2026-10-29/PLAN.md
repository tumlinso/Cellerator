# V100 framework validation and optimization

The user authorizes real-device validation and measured optimization of the
new framework CUDA families on all four available V100s. Root coordinates
integration and Device Scheduler jobs; independent Luna owners cover each
numerical family, runtime/transfer lifetime, Python integration and review.

Correctness compares actual FP16/FP32 device calls and derivatives with
independent host equations or exact integer/set oracles. Boundary cases include
empty extents, tails, repeated occurrences, signed weights, invalid metadata,
capacity, explicit streams, exact parameter revisions and asynchronous owners.
Kernel families are isolated across UUID-selected exclusive device grants.

After correctness, Compute Sanitizer covers memory, initialization, races and
synchronization. Relevant performance comparisons use canonical Autobench,
captured source scope, quiescence and matched inputs. Candidate changes require
independent correctness and measured benefit; resident kernel costs and setup,
conversion, transfer and end-to-end costs retain separate labels. No biological
validity or universal speedup follows from these fixtures.

Project Control's next_task returned workflow_task_already_active; existing
claims are preserved. Pre-existing missing highway/htslib worktree directories
are outside this scope. No GPU is launched outside Scheduler authority, and no
existing benchmark registration or scheduler record is altered.

Guidance: CUDA SKILL.md, host-execution/native/Volta routing, KDA agent entry
and scoped owner drafts. Source and binary hashes, device identity, numerical
tolerances, sanitizer logs and cleanup receipts accompany run evidence.
