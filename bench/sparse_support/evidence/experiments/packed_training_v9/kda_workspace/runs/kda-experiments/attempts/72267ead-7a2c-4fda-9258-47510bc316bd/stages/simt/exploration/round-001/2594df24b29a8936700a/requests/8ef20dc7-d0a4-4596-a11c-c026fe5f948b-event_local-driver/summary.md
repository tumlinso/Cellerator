# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.985600 ms, packed 1.054720 ms (paired SIMT/packed 0.935, 95% bootstrap [0.9257115233874624, 0.9433317280492568]); synchronized wall SIMT 0.993542 ms, packed 1.062710 ms (ratio 0.935).
- training: CUDA event SIMT 4.174336 ms, packed 4.429824 ms (paired SIMT/packed 0.940, 95% bootstrap [0.8531741155958604, 1.0275862076227151]); synchronized wall SIMT 4.185421 ms, packed 4.440189 ms (ratio 0.941).
- Cold setup: SIMT 514.515 ms; packed 471.245 ms including shared geometry compilation; packed extra -43.270 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -43.269693000183906, "training_step_saving_ms": -0.2554880380630493}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -43.269693000183906, "training_step_saving_ms": -0.2547684998717159}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
