# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.955136 ms, packed 1.027840 ms (paired SIMT/packed 0.929, 95% bootstrap [0.9255371918135773, 0.9329692242739476]); synchronized wall SIMT 0.962589 ms, packed 1.035704 ms (ratio 0.929).
- training: CUDA event SIMT 3.791872 ms, packed 3.913984 ms (paired SIMT/packed 0.968, 95% bootstrap [0.9467528998595648, 0.9887794263280404]); synchronized wall SIMT 3.799317 ms, packed 3.921835 ms (ratio 0.968).
- Cold setup: SIMT 519.119 ms; packed 490.158 ms including shared geometry compilation; packed extra -28.961 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -28.960939998796675, "training_step_saving_ms": -0.12211203575134277}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -28.960939998796675, "training_step_saving_ms": -0.12251775024196832}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
