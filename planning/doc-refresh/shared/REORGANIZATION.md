# Bounded source and documentation movement

The user permits internal reorganization and does not require preserving hypothetical external consumers. That does not authorize silent semantic changes or breaking the actual three-repository program.

Default: documentation relocation/consolidation and one coherent mechanical source cluster per repository, only if needed. Keep the existing public/core layouts that already help. Avoid wholesale header renaming, converting extensions, C++ module migration, code formatting, allocator changes and test-system replacement.

Before a source move: inspect definitions/callers/build/install references, active task scopes, submodule boundaries and embedded source hashes. Record old→new path, intended responsibility, exact consumers and validation. After: repair includes, exports, CMake, Python imports, examples and living documentation; run affected checks. Keep immutable historical receipts unchanged and add a resolution map for old paths. Existing tests that enforce old README headings should be redirected to the appropriate new status/evidence page, not deleted wholesale or used to force old prose back into README.

Generated sources/indexes and Todo projections are regenerated using their owner. Never hand-edit .ctxpp or .todo-orchestrator. Do not move a managed workflow block in AGENTS or discard rules that are still operational. Put only genuinely durable scientific/implementation constraints in the new brief AGENTS file; relocate rationale into linked design/development pages.

An active earlier program that touches the same scope must reach a safe handoff before movement. An old run marked active without active claims is not, alone, proof that the repository is locked. Resolve factual scope through current tools.

Archive implies preservation and a visible date/replacement link. “Obsolete” means reviewed and no longer authoritative, not “safe to delete because a filename looks old.” No automatic deletion script is included.

If a useful move is outside the provided local task paths, either leave it in place with a clear index or amend only that task's explicit scope after checking ownership. Do not claim the entire repository or follow a symlink into a sibling checkout to make a cleanup gate pass. In particular, non-live root plan files can remain indexed rather than being moved as a prerequisite.
