# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.002752 ms, packed 1.051136 ms (paired SIMT/packed 0.954, 95% bootstrap [0.9405458108468897, 0.9673806995189111]); synchronized wall SIMT 1.010774 ms, packed 1.059250 ms (ratio 0.954).
- training: CUDA event SIMT 3.790336 ms, packed 3.928320 ms (paired SIMT/packed 0.963, 95% bootstrap [0.8779872230012439, 1.0486368992310897]); synchronized wall SIMT 3.798191 ms, packed 3.936047 ms (ratio 0.963).
- Cold setup: SIMT 513.257 ms; packed 468.367 ms including shared geometry compilation; packed extra -44.891 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -44.89077099788119, "training_step_saving_ms": -0.137984037399292}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -44.89077099788119, "training_step_saving_ms": -0.13785574992652982}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
