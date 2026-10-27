# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.016320 ms, packed 1.083904 ms (paired SIMT/packed 0.938, 95% bootstrap [0.9271553526788513, 0.9486683383118504]); synchronized wall SIMT 1.024189 ms, packed 1.092191 ms (ratio 0.938).
- training: CUDA event SIMT 4.106496 ms, packed 4.170240 ms (paired SIMT/packed 0.983, 95% bootstrap [0.9175101377365349, 1.0480562889316656]); synchronized wall SIMT 4.114387 ms, packed 4.178796 ms (ratio 0.983).
- Cold setup: SIMT 514.822 ms; packed 471.569 ms including shared geometry compilation; packed extra -43.253 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -43.2529629979399, "training_step_saving_ms": -0.0637439489364624}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -43.2529629979399, "training_step_saving_ms": -0.06440949891839409}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
