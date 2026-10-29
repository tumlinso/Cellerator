# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.887808 ms, packed 0.951296 ms (paired SIMT/packed 0.933, 95% bootstrap [0.9256865917826332, 0.9408284512791533]); synchronized wall SIMT 0.896070 ms, packed 0.959163 ms (ratio 0.934).
- training: CUDA event SIMT 3.977480 ms, packed 4.283656 ms (paired SIMT/packed 0.927, 95% bootstrap [0.8477162130806698, 1.0053667501892614]); synchronized wall SIMT 3.985522 ms, packed 4.292294 ms (ratio 0.927).
- Cold setup: SIMT 520.965 ms; packed 474.591 ms including shared geometry compilation; packed extra -46.373 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -46.373467997909756, "training_step_saving_ms": -0.30617618560791016}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -46.373467997909756, "training_step_saving_ms": -0.3067725001528743}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
