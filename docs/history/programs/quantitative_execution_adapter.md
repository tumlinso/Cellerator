# CE-LIVE quantitative execution adapter

CE-LIVE-24 binds the checksum-pinned computational fixture to the native
Cellerator execution identities without adding an H5AD reader to Cellerator.
The extraction tool from CE-LIVE-12 remains responsible for validating H5AD
and producing the local array artifact. This adapter begins with already
validated offsets, indices, and two mutable value arrays.

## Logical relation and physical view

The logical relation is fixed as:

```text
feature/gene source -> cell/row destination
```

`destination_row_csr_view` stores one physical CSR row per destination cell;
its column indices identify source features. Physical row orientation does not
reverse the logical relation. Transpose is not implicit in this adapter and
would require the explicit transpose projection/value-position map defined by
the execution contracts.

`bind_quantitative_fixture` interns the fixture's persistent domain, order,
geometry, partition, structure, and projection identities through the existing
identity registry. The checksum-pinned PBMC3K manifest identities are exposed
by `pbmc3k_quantitative_v1_identities`. SHA-256 fixture identities are narrowed
to the persistent ABI's 128-bit width by taking the first 16 digest bytes in
network byte order; this rule is explicit and independent of host endianness.

The resulting `relation_structure` is immutable. Generation 1 (validated
stored values) and generation 2 (deterministic numerical stress values) are
separate `value_plane` bindings over that same structure. Changing a values
pointer or generation does not change topology, projection, domains, or order.

## Independent quantitative check

`tests/live/quantitative_relation_test.cu` uses the tiny committed schema
fixture with identities derived specifically for that fixture. It does not
mislabel the tiny support as PBMC3K. The test:

- resolves the source domain as features and destination domain as cells;
- validates both value generations and rejects a stale generation;
- creates deterministic dense operands for `N = 1, 16, 17, 31, 32, 48, 64`;
- evaluates the destination-row projection on CUDA; and
- compares every output against an independent host coordinate expansion.

The committed CUDA controller command is:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/runtime_fixture/cuda_controller.json --json
```

This is computational correctness evidence only. The fixture carries no donor,
sample, chemistry, species, normalization, comparison, or biological
interpretation claim.
