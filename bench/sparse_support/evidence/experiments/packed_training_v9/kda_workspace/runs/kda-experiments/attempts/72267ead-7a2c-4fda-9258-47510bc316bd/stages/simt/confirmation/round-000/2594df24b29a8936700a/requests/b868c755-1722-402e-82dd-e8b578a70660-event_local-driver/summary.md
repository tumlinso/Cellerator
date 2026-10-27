# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.042432 ms, packed 1.146368 ms (paired SIMT/packed 0.909, 95% bootstrap [0.8905206950890663, 0.9282832564814791]); synchronized wall SIMT 1.049542 ms, packed 1.154328 ms (ratio 0.909).
- training: CUDA event SIMT 4.275968 ms, packed 4.522240 ms (paired SIMT/packed 0.944, 95% bootstrap [0.8655160684701724, 1.0220328115919552]); synchronized wall SIMT 4.284626 ms, packed 4.529739 ms (ratio 0.944).
- Cold setup: SIMT 512.636 ms; packed 471.253 ms including shared geometry compilation; packed extra -41.383 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -41.38267600137624, "training_step_saving_ms": -0.24627184867858887}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -41.38267600137624, "training_step_saving_ms": -0.2451130003464641}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
