# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.962560 ms, packed 1.044821 ms (paired SIMT/packed 0.920, 95% bootstrap [0.9076821521688332, 0.9240505985500281]); synchronized wall SIMT 0.968083 ms, packed 1.050304 ms (ratio 0.920).
- training: CUDA event SIMT 4.956160 ms, packed 5.324800 ms (paired SIMT/packed 0.979, 95% bootstrap [0.9658055282347148, 0.9899983898093359]); synchronized wall SIMT 4.964223 ms, packed 5.334144 ms (ratio 0.979).
- Preparation: common CUDA/Torch initialization 239.492 ms; fixture owner setup 519.722 ms; CE-GEO host compilation 0.167 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 471.078 ms (p10-p90 467.107-496.115); packed median 472.313 ms (p10-p90 467.826-508.298); packed extra including CE-GEO host compile 1.402 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0822613239288329, "packed_extra_cold_setup_ms": 1.401958001224557, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0822213332867251, "packed_extra_cold_setup_ms": 1.401958001224557, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.3686399459838867, "packed_extra_cold_setup_ms": 1.401958001224557, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.36992099921917543, "packed_extra_cold_setup_ms": 1.401958001224557, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
