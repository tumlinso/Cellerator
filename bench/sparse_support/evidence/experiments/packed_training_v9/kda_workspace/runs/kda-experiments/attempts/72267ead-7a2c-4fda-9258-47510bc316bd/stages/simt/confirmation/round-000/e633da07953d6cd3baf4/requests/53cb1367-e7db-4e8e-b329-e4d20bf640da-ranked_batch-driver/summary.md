# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.998912 ms, packed 1.089280 ms (paired SIMT/packed 0.918, 95% bootstrap [0.8807675179007843, 0.9554695031285633]); synchronized wall SIMT 1.007229 ms, packed 1.098182 ms (ratio 0.918).
- training: CUDA event SIMT 4.015104 ms, packed 4.333824 ms (paired SIMT/packed 0.924, 95% bootstrap [0.8261239383290172, 1.0213817705299941]); synchronized wall SIMT 4.023035 ms, packed 4.342038 ms (ratio 0.924).
- Cold setup: SIMT 514.803 ms; packed 469.432 ms including shared geometry compilation; packed extra -45.371 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -45.37102700123796, "training_step_saving_ms": -0.3187199831008911}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -45.37102700123796, "training_step_saving_ms": -0.3190027500750148}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
