# Native NN CUDA composition qualification

The bounded C++ composition executable passed on the admitted Tesla V100-SXM2-16GB (`GPU-d745da9c-7649-8334-41f1-483afa6f3206`, compute capability 7.0). The normal run exited 0. The separate Compute Sanitizer memcheck run also exited 0 and reported **0 errors**. Device Scheduler verified cleanup for both attempts.

The executable exercised three all-or-fail composition checks: prepared FP16 CSR attention against the ATen implementation for forward values and parameter/input VJPs; a frozen shared model composition through feature encoding, relation attention, interaction programs, gated interaction, feature readout, target alignment, and explicit supervision while preserving finite input gradients; and overlapping-program forward outputs and VJPs. The program prints one success line only after all three checks return. It has no skip path, and both admitted runs reached that success line.

| Run | Device Scheduler job / attempt | Result | Cleanup |
| --- | --- | --- | --- |
| Correctness | `fec15949-d928-4bc3-bbc9-09782543f900` / `de7c772c-aef7-4dc8-868c-c6ed9da62041` | Exit 0; all composition checks passed | Verified |
| Compute Sanitizer memcheck | `79cdb198-7544-40dd-9753-2fea4804a06e` / `ad8b9480-a444-447c-8d07-ac8f4cd832f3` | Exit 0; error summary: 0 | Verified |

The two job specifications requested the same UUID-selected GPU exclusively, 2 GiB admission accounting, and a two-sample idle/quiescence precondition. The Scheduler receipts record that precondition as verified immediately before each attempt and identify the actual device as UUID `GPU-d745da9c-7649-8334-41f1-483afa6f3206`. The memcheck command was `/usr/local/cuda/compute-sanitizer/compute-sanitizer --tool memcheck --error-exitcode 86 /tmp/cellerator-native-nn-cuda/native_nn_cuda_models_test --device`.

Before execution, `cuobjdump --list-elf` reported three embedded `sm_70` cubins. This inspection did not inspect PTX or SASS. The executable SHA256 was `935b8d73174784eeceb78ce12eef7076c14b281d56302f22bda4612773ef79cd`; the tested source `tests/nn/native/cuda_models_test.cc` SHA256 was `bb752ddee3724df654cef31cff3cbd21529d85d3065fb93dc5cc1445b7df7c91`. The binary and source hashes were rechecked after both runs. Build inputs, linked archive hashes, and the source fingerprint are recorded in [models-identity.json](evidence/models-identity.json).

Scheduler submission, status, and output receipts are retained at [models-submission.json](evidence/models-submission.json), [models-status.json](evidence/models-status.json), [models-logs.json](evidence/models-logs.json), [models-memcheck-submission.json](evidence/models-memcheck-submission.json), [models-memcheck-status.json](evidence/models-memcheck-status.json), and [models-memcheck-logs.json](evidence/models-memcheck-logs.json). The ELF listing is [models-elf.txt](evidence/models-elf.txt).

This qualifies the tested model compositions on one V100 and the recorded shapes/data. It establishes neither broad biological validity nor performance. The test source does not set an explicit random seed, so generated tensors are not a source-pinned fixture.

After the executable was captured and qualified, the shared workspace's `cmake/CelleratorNN.cmake` changed as build integration continued. Its pre-run capture hash is retained separately from the current workspace hash in `models-identity.json`. The Scheduler jobs used the unchanged executable hash recorded above; this qualification is not a rebuild of that later CMake revision.
