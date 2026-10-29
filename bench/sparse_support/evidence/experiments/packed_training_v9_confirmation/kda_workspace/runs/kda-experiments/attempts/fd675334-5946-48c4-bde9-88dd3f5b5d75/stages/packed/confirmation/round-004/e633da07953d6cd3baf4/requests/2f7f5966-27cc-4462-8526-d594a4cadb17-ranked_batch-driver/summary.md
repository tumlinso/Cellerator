# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.924331 ms, packed 1.002496 ms (paired SIMT/packed 0.922, 95% bootstrap [0.9189649476915877, 0.9258373110558592]); synchronized wall SIMT 0.929587 ms, packed 1.007571 ms (ratio 0.923).
- training: CUDA event SIMT 7.341398 ms, packed 7.529472 ms (paired SIMT/packed 0.974, 95% bootstrap [0.9718934162941306, 0.9762787106947326]); synchronized wall SIMT 7.353615 ms, packed 7.542096 ms (ratio 0.974).
- Preparation: common CUDA/Torch initialization 212.981 ms; fixture owner setup 512.484 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 470.088 ms (p10-p90 467.879-480.771); packed median 468.893 ms (p10-p90 463.018-469.926); packed extra including CE-GEO host compile -1.053 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0781653722127279, "packed_extra_cold_setup_ms": -1.053374999173684, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07798433337787469, "packed_extra_cold_setup_ms": -1.053374999173684, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18807411193847656, "packed_extra_cold_setup_ms": -1.053374999173684, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18848033384225982, "packed_extra_cold_setup_ms": -1.053374999173684, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
