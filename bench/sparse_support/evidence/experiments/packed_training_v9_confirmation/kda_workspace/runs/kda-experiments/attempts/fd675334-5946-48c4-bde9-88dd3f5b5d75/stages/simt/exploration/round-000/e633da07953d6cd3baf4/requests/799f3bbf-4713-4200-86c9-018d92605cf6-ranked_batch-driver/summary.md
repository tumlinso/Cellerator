# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.934912 ms, packed 1.015467 ms (paired SIMT/packed 0.919, 95% bootstrap [0.9093022568637079, 0.9235947170092854]); synchronized wall SIMT 0.940085 ms, packed 1.020581 ms (ratio 0.919).
- training: CUDA event SIMT 5.384875 ms, packed 5.499563 ms (paired SIMT/packed 0.981, 95% bootstrap [0.9710145412661215, 0.9870250868099587]); synchronized wall SIMT 5.394097 ms, packed 5.508810 ms (ratio 0.981).
- Preparation: common CUDA/Torch initialization 207.966 ms; fixture owner setup 512.372 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 470.560 ms (p10-p90 465.994-471.058); packed median 471.039 ms (p10-p90 469.374-484.294); packed extra including CE-GEO host compile 0.620 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08055472373962402, "packed_extra_cold_setup_ms": 0.620410999545129, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08049566531553864, "packed_extra_cold_setup_ms": 0.620410999545129, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.11468760172525982, "packed_extra_cold_setup_ms": 0.620410999545129, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.11471333345980383, "packed_extra_cold_setup_ms": 0.620410999545129, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
