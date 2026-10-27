# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.909312 ms, packed 0.982016 ms (paired SIMT/packed 0.926, 95% bootstrap [0.9259645309594643, 0.9259645309594643]); synchronized wall SIMT 0.925580 ms, packed 0.999308 ms (ratio 0.926).
- training: CUDA event SIMT 3.343360 ms, packed 3.448832 ms (paired SIMT/packed 0.969, 95% bootstrap [0.9694180270921402, 0.9694180270921402]); synchronized wall SIMT 3.385159 ms, packed 3.465939 ms (ratio 0.977).
- Cold setup: SIMT 521.893 ms; packed 469.107 ms including shared geometry compilation; packed extra -52.787 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -52.78677400201559, "training_step_saving_ms": -0.10547208786010742}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -52.78677400201559, "training_step_saving_ms": -0.08078000246314332}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
