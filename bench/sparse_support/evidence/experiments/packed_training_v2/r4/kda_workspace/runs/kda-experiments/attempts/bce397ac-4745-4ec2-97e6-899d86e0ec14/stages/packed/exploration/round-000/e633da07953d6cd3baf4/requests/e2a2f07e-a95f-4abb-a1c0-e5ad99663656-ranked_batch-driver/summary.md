# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.083392 ms, packed 1.148160 ms (paired SIMT/packed 0.944, 95% bootstrap [0.9232827208824883, 0.9638875844283535]); synchronized wall SIMT 1.092419 ms, packed 1.156930 ms (ratio 0.944).
- training: CUDA event SIMT 4.068864 ms, packed 4.217856 ms (paired SIMT/packed 0.963, 95% bootstrap [0.8919253513908802, 1.0348075296249175]); synchronized wall SIMT 4.077446 ms, packed 4.226164 ms (ratio 0.964).
- Cold setup: SIMT 514.358 ms; packed 465.294 ms including shared geometry compilation; packed extra -49.064 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -49.06417299571331, "training_step_saving_ms": -0.14899182319641113}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -49.06417299571331, "training_step_saving_ms": -0.14871875009703217}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
