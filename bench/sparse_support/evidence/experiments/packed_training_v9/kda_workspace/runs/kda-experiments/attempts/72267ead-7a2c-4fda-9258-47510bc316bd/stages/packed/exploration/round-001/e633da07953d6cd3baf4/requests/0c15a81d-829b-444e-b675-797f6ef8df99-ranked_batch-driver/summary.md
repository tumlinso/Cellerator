# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.966144 ms, packed 1.043456 ms (paired SIMT/packed 0.926, 95% bootstrap [0.9241378728960369, 0.9276637515799425]); synchronized wall SIMT 0.974328 ms, packed 1.051855 ms (ratio 0.926).
- training: CUDA event SIMT 3.598848 ms, packed 3.980544 ms (paired SIMT/packed 0.900, 95% bootstrap [0.7904175234223512, 1.0092855521167612]); synchronized wall SIMT 3.606560 ms, packed 3.988506 ms (ratio 0.900).
- Cold setup: SIMT 517.829 ms; packed 470.743 ms including shared geometry compilation; packed extra -47.085 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -47.085234997211955, "training_step_saving_ms": -0.3816962242126465}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -47.085234997211955, "training_step_saving_ms": -0.38194549961190205}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
