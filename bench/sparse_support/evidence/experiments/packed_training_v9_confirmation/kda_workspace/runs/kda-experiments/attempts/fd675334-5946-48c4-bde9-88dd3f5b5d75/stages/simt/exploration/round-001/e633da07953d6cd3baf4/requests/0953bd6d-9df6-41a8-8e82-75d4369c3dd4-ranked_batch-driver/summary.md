# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.944128 ms, packed 1.028096 ms (paired SIMT/packed 0.918, 95% bootstrap [0.91500668537815, 0.9211285634976344]); synchronized wall SIMT 0.949258 ms, packed 1.033240 ms (ratio 0.918).
- training: CUDA event SIMT 7.445845 ms, packed 7.629824 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9678177623648349, 0.9919397264621336]); synchronized wall SIMT 7.458293 ms, packed 7.643556 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 212.045 ms; fixture owner setup 511.041 ms; CE-GEO host compilation 0.143 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 495.208 ms (p10-p90 476.350-510.191); packed median 490.893 ms (p10-p90 472.084-505.696); packed extra including CE-GEO host compile -4.173 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08396792411804199, "packed_extra_cold_setup_ms": -4.1727940006239805, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08398233452074533, "packed_extra_cold_setup_ms": -4.1727940006239805, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18397839864095022, "packed_extra_cold_setup_ms": -4.1727940006239805, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1852623342225943, "packed_extra_cold_setup_ms": -4.1727940006239805, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
