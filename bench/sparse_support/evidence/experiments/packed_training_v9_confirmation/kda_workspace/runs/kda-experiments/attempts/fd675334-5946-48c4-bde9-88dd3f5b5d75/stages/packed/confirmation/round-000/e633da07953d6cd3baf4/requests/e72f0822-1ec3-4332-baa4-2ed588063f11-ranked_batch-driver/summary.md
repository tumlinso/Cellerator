# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.940373 ms, packed 1.018539 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9159999819944304, 0.925901181265806]); synchronized wall SIMT 0.945455 ms, packed 1.024000 ms (ratio 0.923).
- training: CUDA event SIMT 6.185643 ms, packed 6.135467 ms (paired SIMT/packed 0.985, 95% bootstrap [0.9704557084797124, 0.9963928106041035]); synchronized wall SIMT 6.198320 ms, packed 6.147161 ms (ratio 0.985).
- Preparation: common CUDA/Torch initialization 216.886 ms; fixture owner setup 512.592 ms; CE-GEO host compilation 0.178 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.401 ms (p10-p90 464.765-479.542); packed median 467.046 ms (p10-p90 465.028-470.769); packed extra including CE-GEO host compile -2.177 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07816529273986828, "packed_extra_cold_setup_ms": -2.1766659992863424, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07854500048172974, "packed_extra_cold_setup_ms": -2.1766659992863424, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": 0.050175984700521425, "packed_extra_cold_setup_ms": -2.1766659992863424, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": 0.05115866709578132, "packed_extra_cold_setup_ms": -2.1766659992863424, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
