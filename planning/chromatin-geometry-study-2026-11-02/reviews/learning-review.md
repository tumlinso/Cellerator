# Independent learning-path review

Reviewed 2026-11-02 against the approved study contract in
`bench/chromatin_geometry/docs/draft.md`. This is an initial source review of
the isolated learning paths, not acceptance of the complete study or of GPU
execution. All paths in this record are relative to the repository root.

## Finding: failed STE forward can still commit an update

`bench/chromatin_geometry/learning/ste.cu:57` records a nonfinite candidate
score in the device status word. Nevertheless, `ste_forward` publishes a valid
host snapshot after enqueue (`:149`), while neither `ste_backward` (`:154`)
nor `ste_sgd_update` (`:187`) receives or checks that status. A caller can
enqueue forward, backward, and SGD on one stream before reading the status:
the nonfinite score yields a nonfinite saved probability and gradient, and
SGD writes a nonfinite parameter. A later host check detects the failure but
does not undo the mutation. The existing CUDA test checks that the status is
set, but does not enqueue and check the failed update (`ste_tests.cu:178`).

This is a material failure-propagation risk for the study's native training
chain. Either guard the device-side update on the forward status or establish
and enforce a host status check before any update. Add an admitted test with a
nonfinite candidate that verifies the failure is reported and that the
training parameters remain usable and unchanged. The teacher's update has a
separate guarded commit (`teacher.cu:67`). No GPU reproduction was run for
this source-based finding; the data dependency is visible in the kernels.

## Coverage and limits

Source inspection found the stated sigmoid/admission and analytic STE
estimator, including a derivative mask independent of forward admission. The
teacher uses eight independent logits, native BCE and SGD, threshold hardening,
local LUT composition, and explicit revision checks. No other concrete
arithmetic or circuit defect was identified in the inspected learning code.
This is not evidence that the complete producer-to-consumer-to-update chain is
correct; combined fixtures and all GPU correctness and sanitizer jobs remain
required. The teacher test source includes all 256 LUTs and all eight input
rows, but that coverage had not been executed on an admitted GPU at review.

Host-only checks: the existing `/tmp/cgs-contract-build-129` CTest reported
3/3 passing; its STE library postdated the reviewed STE source, but its teacher
library predated the reviewed teacher source, so its teacher result is not
attributed to the current source. A fresh one-core CUDA 12.9 SM70 compilation
of `teacher.cu` and `teacher_host_test.cc` into `/tmp/cgs-review-teacher`,
followed by running that no-device host executable, passed. Its binary SHA256
was `b792cd7dc02ddeeb0f44e1e64aadb5009eaba3377c62e3c7ad4adb490d0f868d`.
No GPU was used by this reviewer.

Reviewed source SHA256:

| File | SHA256 |
| --- | --- |
| `learning/ste.hh` | `991364ac26ac18a37d8dd8002e2e3f91f3ce70396e3e297f4e48bdceed2795d4` |
| `learning/ste.cu` | `6f76c15a442da0cf9be012b7e03e06de04ffcda9956d4507cf4ac523d9ff17d9` |
| `learning/ste_host_tests.cc` | `1a43b4d0ec6e00432c69d05f1d6760d18a46b6eb0d62da31d6338e797e88f90a` |
| `learning/ste_tests.cu` | `b2cb8201e2deb1194f480dd0a3b4a394142f494bb56b9d4ed12db955e4b3d996` |
| `learning/teacher.hh` | `e34931250c4e6b2747f8f2df6f491be46a399c520a8a07df51c7ff2d9c30b3d7` |
| `learning/teacher.cu` | `06669f1177018e84e9f52aec9665c59ef4f24a2209d5d07c760503abf36e1661` |
| `learning/teacher_host_test.cc` | `31ea494abdc98b312b66c3c1befb1b9a1b8fd13e121a989b69584dd3e57b6f53` |
| `learning/teacher_test.cu` | `5459d36bea5e21b7e8d2108fdbf5bd921976fe5b91fb62d10538b1c303601940` |
| `reference/teacher_reference.hh` | `15eb40548cedda0b9e65ed92808ea6b3c4e0c1101a445a6c1a7a8c3ae5b24f02` |

These file names are relative to `bench/chromatin_geometry`. They identify the
reviewed artifacts; later edits require review against their new identities.
