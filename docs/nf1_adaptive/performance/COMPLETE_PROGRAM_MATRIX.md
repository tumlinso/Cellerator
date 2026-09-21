# Complete-program performance matrix

The measured operation is a retained `local_arithmetic` FP32 multiply JVP
bound through `make_local_device_stage`, then submitted by the ordinary
`prepared_program_v2`. Its second prepared stage is the existing
`dynamic_support_mask_v1`; it receives the JVP output and an exact activity
mask. The host builds the activity map with
`build_exact_projection_v1`, so a sparse row has a real half-active map rather
than a reset that is overwritten by the response stage.

The controller foreground lease is recorded at
`.todo-orchestrator/runtime/background-artifacts/foreground/bb9f0ce2-8cab-4b17-9b09-3543f6ac9696/lease.json`,
with benchmark stdout in its adjacent `foreground.stdout.txt`. It used CUDA
12.9 on the quiescent Tesla V100-SXM2-16GB under the shared NF1A GPU lock.
Every row has seven samples. Cold includes allocation, initial uploads,
preparation, refresh, support upload, program launch/completion and observation
download. Resident retains allocation, preparation and initial uploads but
includes support, response and observation. Amortized spreads one measured setup
across a finite seven-use lifecycle, while each use pays its own fresh refresh,
support, response, host-launch, and observation/download costs.

| Width | Support | Cold median us | Resident median us | Amortized median us |
| ---: | :--- | ---: | ---: | ---: |
| 16 | dense full | 181.909 | 30.772 | 55.988 |
| 16 | exact active half | 177.597 | 29.990 | 54.694 |
| 33 | dense full | 180.407 | 29.793 | 54.750 |
| 33 | exact active half | 174.868 | 29.677 | 53.851 |
| 65 | dense full | 177.138 | 30.528 | 54.969 |
| 65 | exact active half | 176.569 | 30.270 | 54.896 |

At width 33, the median measured phases were 112.594 us allocation, 0.988 us
preparation, 19.456 us initial upload, 4.096 us value refresh, 5.120 us support
upload, 13.312 us response and support completion, 10.288 us host program
submission, and 13.184 us observation/download. The two prepared stages make
two launches per program. Device allocations are 400, 825 and 1,625 bytes for
widths 16, 33 and 65 respectively.

The persistent-mask route does not compact or avoid the JVP kernel, and it
shows no useful support crossover in this small sweep. `local_arithmetic` has
no typed compact response binding, so compact response is explicitly
unsupported rather than reported as a speedup. Its response path is FP32 only;
the FP16 response comparison is also unsupported. No CUDA graph replay is
promoted: the retained V07 contract rejects capture before node creation.
