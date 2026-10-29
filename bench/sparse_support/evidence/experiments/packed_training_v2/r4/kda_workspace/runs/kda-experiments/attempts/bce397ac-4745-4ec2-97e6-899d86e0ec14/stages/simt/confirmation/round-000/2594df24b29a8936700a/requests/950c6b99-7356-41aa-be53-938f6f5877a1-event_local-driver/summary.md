# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.845312 ms, packed 0.919040 ms (paired SIMT/packed 0.921, 95% bootstrap [0.8786902969984218, 0.964099590847666]); synchronized wall SIMT 0.853490 ms, packed 0.926136 ms (ratio 0.923).
- training: CUDA event SIMT 3.273984 ms, packed 3.633408 ms (paired SIMT/packed 0.898, 95% bootstrap [0.8362877702073618, 0.9592245693847132]); synchronized wall SIMT 3.281716 ms, packed 3.641088 ms (ratio 0.898).
- Cold setup: SIMT 528.281 ms; packed 468.951 ms including shared geometry compilation; packed extra -59.331 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -59.33081899638637, "training_step_saving_ms": -0.3594241142272949}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -59.33081899638637, "training_step_saving_ms": -0.3593720002754708}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
