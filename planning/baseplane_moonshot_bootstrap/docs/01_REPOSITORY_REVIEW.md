# Review of the inspected Baseplane state

## Identity and scope

Read-only Project Control inspection on 2 October 2026 observed Baseplane `main` at `f4e9986607becd8528f42c36f0890bd7dd8f4a0d`, clean, TODO revision 216. The old `compat-v2` run remains active as a record; `BITOP-00` and unfinished roadmap tasks are explicitly blocked. No active claims were observed. The observed-state JSON is evidence, not an automatically reusable authorization precondition.

This was a focused source, architecture, instruction and planning review. `evidence/repository-sources.json` records the exact paths/ranges read. It was not an exhaustive line-by-line audit, and no tests were run against the remote repository. Source and ledger observations are distinguished from historical benchmark reports.

## What is already useful

The exact-sequence substrate is real. Current CMake links the allowed-motif, scalar, validity and predicate-plan sources, optionally the CUDA and Highway paths. The library remains `Baseplane::seq`; the existing exact CPU/CUDA tests are useful preservation checks. [R06]

The validity/chunk contract distinguishes packed bases from validity, excludes tail bits, provides a bounded local base-count range (`0x7fffffff`), and carries 64-bit global origin plus local ownership and halo information. Whole-genome work should therefore be built from bounded chunks with explicit global identities, not by passing more than the local limit into the old scanner. [R11]

The eight-byte `sequence_event` core deliberately leaves biological identity, chunk origin, scores, residency and other metadata in descriptors or sidecars. The output contract already separates `total_matches`, `stored_records`, `dropped_records` and `required_capacity`. Preserve that separation in experimental routing queues. [R10]

The version-1 predicate contract is deliberately bounded: 64 nodes, 16 outputs, 16 motifs per motif array and a maximum local span of 32. Its pointer-free representation, verifier, hash and prepared metadata are useful exact-kernel infrastructure. They are not a requirement that every learned hierarchy, operator algebra or routing experiment fit into a 64-node predicate program. [R08–R09]

The shifted packed exact-count CUDA path is an existing asset. Do not replace it simply to make the new architecture stylistically uniform. Its old benchmark comments are historical evidence with their own fixture/output conditions, not fresh performance measurements from this review. [R12]

## Concrete gaps and cautions

A public opcode enum and a prepared-plan structure are not the same thing as a complete executor. The deferred ledger still includes full scalar semantics, grammar, stable/unordered emit fan-in, CUDA families, residency and backend planning. Current source/build observations do not justify saying that all these are already implemented. [R06–R09]

The inspected verifier checks opcode membership and result-kind membership individually. In the complete verifier function read in `src/seq/predicate_plan.cpp`, I did not see opcode-specific result-kind compatibility checks. Treat a successful verification as exactly the guarantee implemented, not as proof that every combination has a typed executable meaning. Add a small targeted fixture when an experiment first depends on this distinction; do not start a broad verifier rewrite speculatively. [R09]

The older compact scanner increments `records_written` to reserve a logical slot before checking capacity; the newer event contract distinguishes logical matches and actual stored records. This is not asserted to be a bug without the full adapter path, but it is a concrete integration trap: do not equate a reservation counter with valid buffer length. The supplied emitter follows the newer semantic distinction. [R10, R12]

The lab can threshold floating records into planes and regroup them, but those steps lose information under their tested representations. Merely adding a source pointer does not undo the loss inside a learned model. A new architecture must say whether it stores detail, reconstructs it, or accepts approximation. [R13–R15]

## What the previous CUDA lab actually taught

**BitLift:** the tested histogram-like Boolean-to-floating summary lost arrangement, and thresholding discarded magnitude. The thread implementation beat the warp implementation in the reported packed fixture. A successor should change the information retained or the work organization—not repeat the same minterm histogram and assume that a warp is inherently better. [R13–R15]

**CarryFold:** ordered affine composition remained an interesting restricted operator, but CUB was the faster executor in the reported scan fixture. The mathematical idea survived independently of the custom shuffle scan. New monomial/block-monomial effects extend the model class; their first executor may still be CUB. [R13–R15]

**Rethread:** compacting work into a warp-cooperative path was slower than the simple thread path across the tested densities. Skipping richer work also changed nonzero synthetic outputs. A successor must explain why its objects have enough useful shared work, and what a skipped result means. “Branchless plus compaction” is not a performance argument. [R13]

**Rendezvous:** the tested packet was assembled with oracle candidate information; local grouping also missed directed peers crossing warp boundaries. A real successor must construct a whole-input candidate directory and account for long posting lists, collisions, movement and missed candidates. This is why E17 is a central foundation for nonlocal experiments rather than an optional decoration. [R13]

All four lab weights/keys were synthetic and untrained. The lab's published checks and sanitizer receipts belong to those historical experiments. They were not rerun here and do not validate the new seeds. [R05, R13]

## Backlog state and replacement

The old roadmap contains **33 records: 8 done, 24 blocked including the root, and 1 superseded**. Every record is mapped in `machine/old-to-new.json` and CSV. Completed tasks remain completed history; the workshop reuses their outputs. Pending execution is integrated into coarse new outcomes or explicitly retained as a later qualification obligation.

`CE-BITOP-40` is already superseded. `CE-BITOP-44` nevertheless retains a dependency on it, and the old adapter checkpoint cannot serve as a current integration gate. The successor does not resurrect `DeviceMathContext` or announce that checkpoint reached. Actual production integration must inspect the Cellerator-owned `CE-ARCH-40` and current shared interface. [R07; Project Control architecture-context observation]

The package also revises the *policy* that would otherwise deadlock the new work: the isolated experimental exception expands beyond `experiments/cuda_lab`; learned/float/runtime experiments become authorized without evidence of a speedup; production promotion and broad testing remain separate. Importing task records alone does not silently change those invariants. Policy and scheduling must be reconciled at cutover.

## Preserve, change, postpone

Preserve exact representations, validity, source identity, counters, existing build targets, successful outcomes and negative lab evidence. Change the assumption that all future work must pass through a production-grade BitOp v1 fan-in or a top-two mechanism selection. Postpone comprehensive executor coverage, benchmarking, public ABI freeze, full training evaluation, Torch production binding, persistence and multi-GPU integration.

The result is not abandonment of BitOp. It puts BitOp back in its intended role: the precise, cheap grounding machinery beneath a much larger representation experiment.
