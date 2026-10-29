# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.851200 ms, packed 0.942848 ms (paired SIMT/packed 0.903, 95% bootstrap [0.8881042641141718, 0.9174809418896697]); synchronized wall SIMT 0.861361 ms, packed 0.950589 ms (ratio 0.906).
- training: CUDA event SIMT 4.040192 ms, packed 4.181760 ms (paired SIMT/packed 0.965, 95% bootstrap [0.8828784431059539, 1.0472507471401364]); synchronized wall SIMT 4.047969 ms, packed 4.189991 ms (ratio 0.965).
- Cold setup: SIMT 523.606 ms; packed 477.278 ms including shared geometry compilation; packed extra -46.329 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -46.32869800116168, "training_step_saving_ms": -0.14156794548034668}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -46.32869800116168, "training_step_saving_ms": -0.14202225156623172}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
