# Native training example target correction

The example's regression target changed from `0.40f` to `0.25f` in
`examples/native_cuda/factorized_interaction_training.cu`. The previous target
could push the FP16 alpha parameter beyond the initial `[-1,1]` CUDA
qualification domain after 200 SGD updates. The change is limited to the
example; the frozen correctness corpus and its tolerances were left unchanged.

An independent standard-library CPU replay models the one-mechanism law, FP16
round-to-nearest storage for parameters/activations/output, FP32 gate/VJP/update
arithmetic, and the example's `learning_rate=0.05`. It does not call the CUDA
implementation, Torch adapter, or benchmark oracle. Results and the script are
in `cpu-training-replay.log` and `cpu_training_replay.py`:

| Steps | Initial output | Final output | Alpha | Bias | Gamma | Absolute target error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.2023926 | 0.2301025 | 0.8974609 | -0.1691895 | 0.8999023 | 0.0198975 |
| 10,000 | 0.2023926 | 0.2491455 | 0.9008789 | 0.0156250 | 0.8999023 | 0.0008545 |

Across each replay, all parameter values remained finite and inside `[-1,1]`.
Observed ranges were alpha `[0.7998047, 0.9008789]`, bias
`[-0.1999512, 0.0156250]`, and gamma fixed at `0.8999023`. This is a CPU
trajectory check for the demonstration, not CUDA correctness evidence.

The example recompiled successfully for SM70 with NVCC 12.9. No executable or
GPU runtime was launched. Source SHA-256 is
`ce44d494431458b4c53dde4330cd2555a942b0ca70d68f04c5da12ca49f54ccb`; rebuilt
binary SHA-256 is
`412f537534e8534eca8d3eb62fa1461beedf027281a89d9b9f4eedafcafe0eb8`.
