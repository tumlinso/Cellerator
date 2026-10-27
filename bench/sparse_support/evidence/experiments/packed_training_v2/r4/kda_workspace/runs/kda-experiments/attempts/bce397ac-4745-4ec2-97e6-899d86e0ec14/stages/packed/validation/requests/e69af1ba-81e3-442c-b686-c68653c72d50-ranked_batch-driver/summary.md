# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.925696 ms, packed 0.985088 ms (paired SIMT/packed 0.940, 95% bootstrap [0.93970896411298, 0.93970896411298]); synchronized wall SIMT 0.941287 ms, packed 1.003207 ms (ratio 0.938).
- training: CUDA event SIMT 3.212288 ms, packed 3.340288 ms (paired SIMT/packed 0.962, 95% bootstrap [0.9616799437951667, 0.9616799437951667]); synchronized wall SIMT 3.228088 ms, packed 3.356888 ms (ratio 0.962).
- Cold setup: SIMT 526.905 ms; packed 475.964 ms including shared geometry compilation; packed extra -50.941 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -50.94056800226099, "training_step_saving_ms": -0.12800002098083496}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -50.94056800226099, "training_step_saving_ms": -0.12879999849246815}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
