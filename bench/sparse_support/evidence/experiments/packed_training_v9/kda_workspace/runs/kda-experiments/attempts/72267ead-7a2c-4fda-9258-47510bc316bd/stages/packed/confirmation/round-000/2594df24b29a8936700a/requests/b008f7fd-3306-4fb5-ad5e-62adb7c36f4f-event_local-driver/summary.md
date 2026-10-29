# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.976128 ms, packed 1.033728 ms (paired SIMT/packed 0.944, 95% bootstrap [0.9322034508149305, 0.9562007426288824]); synchronized wall SIMT 0.983945 ms, packed 1.042077 ms (ratio 0.944).
- training: CUDA event SIMT 4.048128 ms, packed 4.302592 ms (paired SIMT/packed 0.940, 95% bootstrap [0.8424599917437454, 1.0370675810534489]); synchronized wall SIMT 4.056489 ms, packed 4.310996 ms (ratio 0.940).
- Cold setup: SIMT 529.236 ms; packed 479.762 ms including shared geometry compilation; packed extra -49.473 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -49.47348000132479, "training_step_saving_ms": -0.25446391105651855}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -49.47348000132479, "training_step_saving_ms": -0.25450724933762103}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
