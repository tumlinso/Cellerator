# Snapshot restore attempt: v10 (failed, retained for diagnosis)

Scheduler job `4c7f4983-28d3-4c9c-84ec-097f75148c89` ran the frozen v10 Python suite on the admitted GPU1 resource and terminated with exit code 1; cleanup was verified. The suite ran 28 tests and had three checkpoint errors.

All three failures occurred when loading a checkpoint into a fresh module. The binding rejected the restored `edge_ids` array as an incompatible dtype/layout. Reproduction showed that `torch.load` returns a contiguous `numpy.int64` array whose dtype compares equal to native `int64` but is not the same dtype object. The binding used dtype object identity, so in-memory snapshot roundtrips passed while serialized checkpoint restore failed.

This attempt is diagnostic only. It is superseded by v11, which compares dtype values and passes the full suite. The exact scheduler receipt, spec, stderr/stdout, frozen identity, source hashes, and extension hash are preserved here. No binary is copied.
