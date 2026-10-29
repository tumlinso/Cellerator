# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.943445 ms, packed 1.029803 ms (paired SIMT/packed 0.916, 95% bootstrap [0.9103153769194761, 0.9177901899671358]); synchronized wall SIMT 0.949316 ms, packed 1.035830 ms (ratio 0.916).
- training: CUDA event SIMT 6.863872 ms, packed 6.975488 ms (paired SIMT/packed 0.978, 95% bootstrap [0.9561872116156203, 0.9852591481451793]); synchronized wall SIMT 6.876037 ms, packed 6.988155 ms (ratio 0.978).
- Preparation: common CUDA/Torch initialization 218.396 ms; fixture owner setup 507.924 ms; CE-GEO host compilation 0.143 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 470.839 ms (p10-p90 464.536-472.743); packed median 470.079 ms (p10-p90 465.071-492.118); packed extra including CE-GEO host compile -0.617 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08635727564493811, "packed_extra_cold_setup_ms": -0.6174589980219025, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08651433381601237, "packed_extra_cold_setup_ms": -0.6174589980219025, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.11161613464355469, "packed_extra_cold_setup_ms": -0.6174589980219025, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1121179993788246, "packed_extra_cold_setup_ms": -0.6174589980219025, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
