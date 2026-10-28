#!/usr/bin/env python3
"""CPU-only regression for E7 stale-structure decision/report plumbing.

This unit test exercises the pure report predicate only. It does not simulate
or claim native owner restoration, support publication, or GPU correctness.
"""

from e7_relation_runner import stale_reset_guard_evidence


accepted = stale_reset_guard_evidence(
    expected_epoch=1,
    published_epoch=2,
    handle_changed=True,
    rejection="fixed-structure reset rejected: support epoch, handle, owner, or identity changed",
)
assert accepted == {
    "accepted": True,
    "expected_epoch": 1,
    "published_epoch": 2,
    "handle_changed": True,
    "rejection": "fixed-structure reset rejected: support epoch, handle, owner, or identity changed",
}

unchanged_epoch = stale_reset_guard_evidence(
    expected_epoch=1,
    published_epoch=1,
    handle_changed=True,
    rejection="fixed-structure reset rejected: support epoch, handle, owner, or identity changed",
)
assert not unchanged_epoch["accepted"]

wrong_exception = stale_reset_guard_evidence(
    expected_epoch=1,
    published_epoch=2,
    handle_changed=True,
    rejection="unrelated runtime failure",
)
assert not wrong_exception["accepted"]

print("CPU-only stale reset guard reporting checks passed; no native/GPU claim")
