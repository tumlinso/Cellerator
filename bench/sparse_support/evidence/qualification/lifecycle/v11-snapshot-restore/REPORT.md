# Sparse support cold snapshot restore qualification (v11)

## Result and identity

The v11 Torch extension changes NumPy dtype validation from object identity to dtype value equality, allowing a checkpoint's reloaded arrays while keeping exact dtype and layout validation. Frozen identity SHA-256: `887f8a27b2e8c1e858690291401919f7cec6898a401798377c31294326a07a5b`; `_torch` SHA-256: `430646858b79724b2eee8129e5a7ec57df6fa9632d89d798fd03d80b67324c5a`. The source-only hashes and extension hash are recorded in separate JSON files; no binaries are copied into this evidence directory.

GPU1 scheduler job `34027c5b-2a17-4922-9eef-4bbc405340e7` succeeded with exit code 0 and verified cleanup. It ran all 28 Python tests with CUDA tests enabled. The independent CPU NumPy binding suite passed 5/5. The earlier v10 failure is retained in [`v10-snapshot-restore`](../v10-snapshot-restore/).

## Restore contract exercised

Checkpoint restore now reinstates the saved active base/delta partition, structure epoch, revoked edge IDs, promotion evidence, gate policy, and application state. The existing native support manager is replaced with the validated checkpoint snapshot, and the numerical handle is rebound against that published snapshot while retaining its canonical parameter owner and Torch leaf.

- The GPU checkpoint case promoted novel edge 12 at epoch 2, revoked it at epoch 3, and saved a state whose active graph matches the initial graph. Loading into a fresh epoch-1 module restored the exact epoch-3 snapshot and tombstone.
- The test verified that the parameter leaf identity, owner storage address, Adam options/state, and next optimizer update continue unchanged.
- Attempting to promote retired edge 12 again was rejected.
- The CPU binding roundtrip restored the exact base/delta partition after base promotion, retained tombstone 40 and promotion evidence for edges 30 and 40, and rejected revival.
- CPU checks rejected restore while a forward tape was live and rejected an active edge ID with conflicting endpoints; the endpoint-conflict case left the target epoch unchanged.
- The existing full lifecycle test for promotion plus revocation and checkpoint continuation passed.

## Reproduction

The exact scheduler request and environment are in `job_spec_and_result.json`. The scheduled command used the frozen v11 tests:

```sh
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/package:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CELLERATOR_RUN_CUDA_TESTS=1 \
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -u \
-m unittest discover \
-s /home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/tests \
-p 'test_*.py' -v
```

This command is the scheduler job argv, not a direct GPU invocation. GPU1 was exclusively admitted after the configured three-sample quiescence check.

The CPU binding suite was run separately against the same frozen v11 package:

```sh
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/package:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11 \
PYTHONDONTWRITEBYTECODE=1 \
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -m unittest discover \
-s /home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v11/tests \
-p test_indexing_bindings.py -v
```

The Python checkpoint tests cover the pair-specialized Torch application path. They do not establish compatibility with arbitrary external checkpoint schemas or prove concurrent restore while any tape remains live; restore explicitly rejects live tapes.
