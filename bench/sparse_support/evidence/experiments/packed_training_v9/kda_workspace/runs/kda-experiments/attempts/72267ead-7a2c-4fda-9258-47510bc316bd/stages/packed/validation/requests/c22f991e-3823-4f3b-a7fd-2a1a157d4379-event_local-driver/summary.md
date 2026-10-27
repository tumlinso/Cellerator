# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.997376 ms, packed 1.069056 ms (paired SIMT/packed 0.933, 95% bootstrap [0.9329501849487036, 0.9329501849487036]); synchronized wall SIMT 1.013358 ms, packed 1.084830 ms (ratio 0.934).
- training: CUDA event SIMT 3.410944 ms, packed 3.575808 ms (paired SIMT/packed 0.954, 95% bootstrap [0.9538945991959209, 0.9538945991959209]); synchronized wall SIMT 3.426200 ms, packed 3.591266 ms (ratio 0.954).
- Cold setup: SIMT 511.924 ms; packed 471.777 ms including shared geometry compilation; packed extra -40.146 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -40.146495004592, "training_step_saving_ms": -0.16486406326293945}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -40.146495004592, "training_step_saving_ms": -0.16506600150023587}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
