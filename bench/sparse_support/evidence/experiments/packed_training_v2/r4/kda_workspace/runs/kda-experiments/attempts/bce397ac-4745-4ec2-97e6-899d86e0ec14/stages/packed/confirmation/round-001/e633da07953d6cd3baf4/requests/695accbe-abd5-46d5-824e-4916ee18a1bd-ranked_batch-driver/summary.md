# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.898048 ms, packed 0.962560 ms (paired SIMT/packed 0.933, 95% bootstrap [0.9263157810892013, 0.9397849871548833]); synchronized wall SIMT 0.910183 ms, packed 0.971093 ms (ratio 0.937).
- training: CUDA event SIMT 3.741440 ms, packed 3.806208 ms (paired SIMT/packed 0.982, 95% bootstrap [0.9328914716664533, 1.0305533895391688]); synchronized wall SIMT 3.749497 ms, packed 3.810330 ms (ratio 0.983).
- Cold setup: SIMT 571.233 ms; packed 501.833 ms including shared geometry compilation; packed extra -69.400 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -69.39997100562323, "training_step_saving_ms": -0.06476795673370361}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -69.39997100562323, "training_step_saving_ms": -0.06083275093260454}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
