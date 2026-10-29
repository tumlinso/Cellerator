# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.991232 ms, packed 1.055744 ms (paired SIMT/packed 0.939, 95% bootstrap [0.9388942103196815, 0.9388942103196815]); synchronized wall SIMT 1.005994 ms, packed 1.072928 ms (ratio 0.938).
- training: CUDA event SIMT 3.377152 ms, packed 3.402752 ms (paired SIMT/packed 0.992, 95% bootstrap [0.9924766903098728, 0.9924766903098728]); synchronized wall SIMT 3.393440 ms, packed 3.417958 ms (ratio 0.993).
- Cold setup: SIMT 508.698 ms; packed 470.124 ms including shared geometry compilation; packed extra -38.574 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -38.573941001232015, "training_step_saving_ms": -0.025599956512451172}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -38.573941001232015, "training_step_saving_ms": -0.024518001737305894}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
