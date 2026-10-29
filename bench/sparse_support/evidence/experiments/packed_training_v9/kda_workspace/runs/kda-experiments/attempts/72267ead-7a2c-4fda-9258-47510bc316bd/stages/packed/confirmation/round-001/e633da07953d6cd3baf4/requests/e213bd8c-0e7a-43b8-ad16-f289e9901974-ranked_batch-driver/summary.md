# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.019648 ms, packed 1.044992 ms (paired SIMT/packed 0.976, 95% bootstrap [0.9603174465684852, 0.9908035272153791]); synchronized wall SIMT 1.027713 ms, packed 1.052936 ms (ratio 0.976).
- training: CUDA event SIMT 4.284160 ms, packed 4.426240 ms (paired SIMT/packed 0.967, 95% bootstrap [0.9322113534815988, 1.0019204055903341]); synchronized wall SIMT 4.292212 ms, packed 4.434277 ms (ratio 0.967).
- Cold setup: SIMT 519.289 ms; packed 476.536 ms including shared geometry compilation; packed extra -42.753 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -42.75336200589663, "training_step_saving_ms": -0.14208030700683594}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -42.75336200589663, "training_step_saving_ms": -0.14206549985829042}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
