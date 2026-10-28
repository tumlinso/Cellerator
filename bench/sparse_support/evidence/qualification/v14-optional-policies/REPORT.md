# Optional policies and broad intake integration

The v14 Python snapshot adds explicit choices while retaining existing
behavior: full-support or P∪D shared-value evaluation, and current-event or
eligible-history relation inputs. No automatic density dispatcher removes a
choice. The native extension is unchanged from qualified v13.

The immutable snapshot is
`build-sparse-support/evidence/torch-v14-optional-policies/`. Its identity
SHA-256 is
`696b15c916d782666565be0f36c09b4e4958046eaac46350cf137435876a0bc9`;
the extension SHA-256 remains
`51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`.
[package_identity.json](package_identity.json) records source and snapshot
hashes, including the evaluation intake sources used by these tests.

Admitted GPU3 job `9f74f089-f94a-421d-87e3-ed4e9082bb63` succeeded with
exit code 0 and verified cleanup after three idle observations. It passed
**83 tests and two subtests in 9.70 seconds**, followed by the rebuilt native
CUDA workset test. The [scheduler record](job_get.json),
[specification](job_spec.json), and [runner](run_regressions.py) retain the
execution contract. This run includes the existing sparse-support and native
mechanism bindings, broad CPU input tests, and the new optional-policy tests.

The new native relation comparison checks forward, input gradients, complete
shared-parameter gradients, and gate-score gradients against full values. Its
workset contains an admitted primal edge, a closed derivative-only edge with
nonzero scorer gradient, and an unused support edge. Other tests cover duplicate
and empty unions, FP16 input rounding, repeated history occurrences, policy
locking, and legacy checkpoint interpretation. Native consumer validation
continues to own ordinal/identity consistency; the optional value path adds
no per-launch host identity scan or synchronization.

The broad input tests preserve opaque external IDs and rank/event order while
exporting reversible native ID/ordinal maps. They check independent windows,
conditional eligibility, explicit capacities, malformed mapping keys, empty
inputs, raw-bundle loader compatibility, and sparse storage without a dense
cell-by-edge matrix. Intake remains CPU preparation, with explicit adapter
responsibilities before native GPU execution.

An opted-in history stream must resume with
`relation_input="eligible_history"`. Omitting the option requests the existing
`current_event` default and rejects a change after stream progress. Legacy
checkpoints without a policy key retain their original current-event meaning.

These checks establish implementation and gradient behavior. Performance and
learning evidence remain the separate, source-bound
[indexed-value experiment](../../../experiments/indexed_values/evidence/qualification/v13/REPORT.md)
and [event readout experiment](../../../experiments/event_learning_diagnostic/README.md).
Their synthetic results do not eliminate other mechanisms or establish
biological validity. No real-data or v14 performance claim is made here.
