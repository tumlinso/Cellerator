# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.938667 ms, packed 1.011712 ms (paired SIMT/packed 0.922, 95% bootstrap [0.9166666601205958, 0.928010924849242]); synchronized wall SIMT 0.944121 ms, packed 1.016836 ms (ratio 0.923).
- training: CUDA event SIMT 7.369387 ms, packed 7.567701 ms (paired SIMT/packed 0.974, 95% bootstrap [0.972162353543128, 0.9761905461852605]); synchronized wall SIMT 7.382007 ms, packed 7.579850 ms (ratio 0.974).
- Preparation: common CUDA/Torch initialization 217.732 ms; fixture owner setup 514.816 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.689 ms (p10-p90 468.109-479.791); packed median 467.476 ms (p10-p90 467.176-470.883); packed extra including CE-GEO host compile -1.070 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07304533322652185, "packed_extra_cold_setup_ms": -1.070124995749211, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07271466650612024, "packed_extra_cold_setup_ms": -1.070124995749211, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19831466674804688, "packed_extra_cold_setup_ms": -1.070124995749211, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1978423327576211, "packed_extra_cold_setup_ms": -1.070124995749211, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
