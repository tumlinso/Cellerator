# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.014528 ms, packed 1.055744 ms (paired SIMT/packed 0.961, 95% bootstrap [0.9267116701142217, 0.9956097226896187]); synchronized wall SIMT 1.022303 ms, packed 1.063748 ms (ratio 0.961).
- training: CUDA event SIMT 4.124160 ms, packed 4.522240 ms (paired SIMT/packed 0.910, 95% bootstrap [0.8285448812571783, 0.9910674992150824]); synchronized wall SIMT 4.132207 ms, packed 4.530385 ms (ratio 0.910).
- Cold setup: SIMT 524.909 ms; packed 478.560 ms including shared geometry compilation; packed extra -46.349 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -46.34906600040267, "training_step_saving_ms": -0.39807987213134766}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -46.34906600040267, "training_step_saving_ms": -0.3981787494922173}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
