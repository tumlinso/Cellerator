# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.988160 ms, packed 1.075200 ms (paired SIMT/packed 0.917, 95% bootstrap [0.9130158760193824, 0.9222682378198567]); synchronized wall SIMT 0.993437 ms, packed 1.080533 ms (ratio 0.917).
- training: CUDA event SIMT 7.693995 ms, packed 7.899478 ms (paired SIMT/packed 0.974, 95% bootstrap [0.9712737404356606, 0.9774293366092692]); synchronized wall SIMT 7.706471 ms, packed 7.912292 ms (ratio 0.974).
- Preparation: common CUDA/Torch initialization 215.258 ms; fixture owner setup 514.559 ms; CE-GEO host compilation 0.150 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.633 ms (p10-p90 467.486-491.038); packed median 469.052 ms (p10-p90 467.536-469.772); packed extra including CE-GEO host compile 0.569 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08704002698262536, "packed_extra_cold_setup_ms": 0.568519997614203, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08709566706481076, "packed_extra_cold_setup_ms": 0.568519997614203, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.20548311869303415, "packed_extra_cold_setup_ms": 0.568519997614203, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.2058213334142538, "packed_extra_cold_setup_ms": 0.568519997614203, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
