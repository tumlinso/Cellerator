# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.091072 ms, packed 1.192184 ms (paired SIMT/packed 0.915, 95% bootstrap [0.9124960607545272, 0.9178847566399275]); synchronized wall SIMT 1.099503 ms, packed 1.200578 ms (ratio 0.916).
- training: CUDA event SIMT 4.199168 ms, packed 4.397824 ms (paired SIMT/packed 0.955, 95% bootstrap [0.8914338786564097, 1.0177453568352401]); synchronized wall SIMT 4.208239 ms, packed 4.406563 ms (ratio 0.955).
- Cold setup: SIMT 538.414 ms; packed 480.235 ms including shared geometry compilation; packed extra -58.179 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -58.17918200409622, "training_step_saving_ms": -0.19865596294403076}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -58.17918200409622, "training_step_saving_ms": -0.1983242509595584}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
