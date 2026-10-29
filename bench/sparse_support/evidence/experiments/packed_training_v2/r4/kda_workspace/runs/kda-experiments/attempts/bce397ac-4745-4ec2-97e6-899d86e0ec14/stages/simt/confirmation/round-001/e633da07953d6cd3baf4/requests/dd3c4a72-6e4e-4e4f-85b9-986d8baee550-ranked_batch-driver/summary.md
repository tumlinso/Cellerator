# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.944128 ms, packed 1.089792 ms (paired SIMT/packed 0.867, 95% bootstrap [0.8424976673538027, 0.8905731824981945]); synchronized wall SIMT 0.952427 ms, packed 1.098211 ms (ratio 0.867).
- training: CUDA event SIMT 3.716096 ms, packed 3.865088 ms (paired SIMT/packed 0.960, 95% bootstrap [0.8930264117907327, 1.0269674823759]); synchronized wall SIMT 3.724543 ms, packed 3.873659 ms (ratio 0.960).
- Cold setup: SIMT 532.124 ms; packed 502.468 ms including shared geometry compilation; packed extra -29.656 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -29.65577599752578, "training_step_saving_ms": -0.14899194240570068}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -29.65577599752578, "training_step_saving_ms": -0.14911650032445323}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
