# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.940032 ms, packed 1.026016 ms (paired SIMT/packed 0.916, 95% bootstrap [0.9161962465693008, 0.9161962465693008]); synchronized wall SIMT 0.955931 ms, packed 1.043075 ms (ratio 0.916).
- training: CUDA event SIMT 3.517440 ms, packed 3.687424 ms (paired SIMT/packed 0.954, 95% bootstrap [0.9539017302077947, 0.9539017302077947]); synchronized wall SIMT 3.534635 ms, packed 3.703459 ms (ratio 0.954).
- Cold setup: SIMT 564.959 ms; packed 473.844 ms including shared geometry compilation; packed extra -91.115 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -91.11527800268959, "training_step_saving_ms": -0.1699838638305664}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -91.11527800268959, "training_step_saving_ms": -0.16882399722817354}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
