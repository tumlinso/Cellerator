# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.961792 ms, packed 1.036800 ms (paired SIMT/packed 0.928, 95% bootstrap [0.913962662158711, 0.9414682697795754]); synchronized wall SIMT 0.969640 ms, packed 1.046881 ms (ratio 0.926).
- training: CUDA event SIMT 4.103936 ms, packed 4.353024 ms (paired SIMT/packed 0.940, 95% bootstrap [0.8451235462330822, 1.0340120797553707]); synchronized wall SIMT 4.112040 ms, packed 4.361264 ms (ratio 0.940).
- Cold setup: SIMT 511.153 ms; packed 475.482 ms including shared geometry compilation; packed extra -35.671 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -35.67103599925758, "training_step_saving_ms": -0.24908804893493652}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -35.67103599925758, "training_step_saving_ms": -0.2492239991624956}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
