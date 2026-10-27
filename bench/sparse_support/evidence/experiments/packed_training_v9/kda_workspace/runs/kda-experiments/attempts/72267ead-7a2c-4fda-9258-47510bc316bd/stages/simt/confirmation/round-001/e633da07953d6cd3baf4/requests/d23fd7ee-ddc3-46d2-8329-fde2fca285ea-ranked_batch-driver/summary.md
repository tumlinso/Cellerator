# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.998656 ms, packed 1.023744 ms (paired SIMT/packed 0.976, 95% bootstrap [0.9323046328895721, 1.019095505608118]); synchronized wall SIMT 1.008643 ms, packed 1.031676 ms (ratio 0.978).
- training: CUDA event SIMT 4.205568 ms, packed 4.403968 ms (paired SIMT/packed 0.952, 95% bootstrap [0.878257718807997, 1.0262479409569671]); synchronized wall SIMT 4.213323 ms, packed 4.412075 ms (ratio 0.952).
- Cold setup: SIMT 517.454 ms; packed 470.545 ms including shared geometry compilation; packed extra -46.909 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -46.909432003303664, "training_step_saving_ms": -0.19839990139007568}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -46.909432003303664, "training_step_saving_ms": -0.198752749383857}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
