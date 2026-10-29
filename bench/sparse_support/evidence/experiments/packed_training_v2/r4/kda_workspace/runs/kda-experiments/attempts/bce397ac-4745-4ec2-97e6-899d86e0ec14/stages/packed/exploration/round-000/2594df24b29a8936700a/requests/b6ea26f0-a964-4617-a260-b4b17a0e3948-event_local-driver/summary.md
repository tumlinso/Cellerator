# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.073664 ms, packed 1.149440 ms (paired SIMT/packed 0.934, 95% bootstrap [0.9310499150395968, 0.937109662087145]); synchronized wall SIMT 1.082586 ms, packed 1.158063 ms (ratio 0.935).
- training: CUDA event SIMT 4.040448 ms, packed 4.321536 ms (paired SIMT/packed 0.935, 95% bootstrap [0.8299728959971466, 1.0409524725007597]); synchronized wall SIMT 4.049320 ms, packed 4.330198 ms (ratio 0.936).
- Cold setup: SIMT 513.402 ms; packed 466.025 ms including shared geometry compilation; packed extra -47.377 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -47.3769220043323, "training_step_saving_ms": -0.2810877561569214}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -47.3769220043323, "training_step_saving_ms": -0.2808782492138562}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
