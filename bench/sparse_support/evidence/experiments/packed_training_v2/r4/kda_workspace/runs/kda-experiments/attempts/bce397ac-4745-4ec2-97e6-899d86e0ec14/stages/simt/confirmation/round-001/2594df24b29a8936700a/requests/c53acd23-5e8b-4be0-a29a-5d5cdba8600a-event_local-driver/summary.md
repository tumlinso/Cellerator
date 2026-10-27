# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.116928 ms, packed 1.198592 ms (paired SIMT/packed 0.932, 95% bootstrap [0.9308874393922644, 0.9328485782293997]); synchronized wall SIMT 1.125376 ms, packed 1.207244 ms (ratio 0.932).
- training: CUDA event SIMT 4.100864 ms, packed 4.352768 ms (paired SIMT/packed 0.939, 95% bootstrap [0.8854631225206967, 0.9933908123622304]); synchronized wall SIMT 4.109802 ms, packed 4.361548 ms (ratio 0.940).
- Cold setup: SIMT 508.068 ms; packed 473.359 ms including shared geometry compilation; packed extra -34.710 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -34.709659001237014, "training_step_saving_ms": -0.2519040107727051}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -34.709659001237014, "training_step_saving_ms": -0.2517455004635849}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
