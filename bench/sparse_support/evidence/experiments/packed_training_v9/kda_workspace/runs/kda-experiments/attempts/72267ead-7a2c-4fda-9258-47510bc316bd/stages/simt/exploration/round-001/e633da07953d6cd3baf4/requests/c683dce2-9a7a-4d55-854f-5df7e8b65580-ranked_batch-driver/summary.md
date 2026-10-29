# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.999680 ms, packed 1.063424 ms (paired SIMT/packed 0.940, 95% bootstrap [0.921391202528725, 0.9591241171979926]); synchronized wall SIMT 1.007517 ms, packed 1.071720 ms (ratio 0.940).
- training: CUDA event SIMT 4.187648 ms, packed 4.382464 ms (paired SIMT/packed 0.953, 95% bootstrap [0.8876160489350321, 1.0193745914383954]); synchronized wall SIMT 4.198582 ms, packed 4.390281 ms (ratio 0.954).
- Cold setup: SIMT 546.059 ms; packed 473.010 ms including shared geometry compilation; packed extra -73.050 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -73.04961599948001, "training_step_saving_ms": -0.194815993309021}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -73.04961599948001, "training_step_saving_ms": -0.19169824918208178}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
