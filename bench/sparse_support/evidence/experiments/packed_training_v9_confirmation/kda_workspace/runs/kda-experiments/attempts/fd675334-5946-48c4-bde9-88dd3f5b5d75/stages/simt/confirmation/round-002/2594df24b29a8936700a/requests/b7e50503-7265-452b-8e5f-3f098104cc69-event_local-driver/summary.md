# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.997376 ms, packed 1.080661 ms (paired SIMT/packed 0.924, 95% bootstrap [0.9185091984286358, 0.9295641683775969]); synchronized wall SIMT 1.002981 ms, packed 1.086157 ms (ratio 0.924).
- training: CUDA event SIMT 7.590229 ms, packed 7.818923 ms (paired SIMT/packed 0.974, 95% bootstrap [0.9714671212867912, 0.9771132321614722]); synchronized wall SIMT 7.602982 ms, packed 7.831659 ms (ratio 0.974).
- Preparation: common CUDA/Torch initialization 219.606 ms; fixture owner setup 516.476 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 472.417 ms (p10-p90 471.034-495.317); packed median 473.413 ms (p10-p90 471.562-479.827); packed extra including CE-GEO host compile 1.137 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08328533172607422, "packed_extra_cold_setup_ms": 1.1367300030542538, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08317666773412702, "packed_extra_cold_setup_ms": 1.1367300030542538, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.22869364420572946, "packed_extra_cold_setup_ms": 1.1367300030542538, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.2286769995407667, "packed_extra_cold_setup_ms": 1.1367300030542538, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
