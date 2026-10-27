# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.970752 ms, packed 1.050624 ms (paired SIMT/packed 0.925, 95% bootstrap [0.9204842968834521, 0.9328114261673995]); synchronized wall SIMT 0.976256 ms, packed 1.056591 ms (ratio 0.926).
- training: CUDA event SIMT 7.503531 ms, packed 7.692971 ms (paired SIMT/packed 0.977, 95% bootstrap [0.9748991029334287, 0.9809210696498519]); synchronized wall SIMT 7.516460 ms, packed 7.705262 ms (ratio 0.978).
- Preparation: common CUDA/Torch initialization 216.498 ms; fixture owner setup 515.557 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 467.435 ms (p10-p90 466.607-498.639); packed median 468.930 ms (p10-p90 465.715-474.296); packed extra including CE-GEO host compile 1.637 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07987197240193677, "packed_extra_cold_setup_ms": 1.6371579986298457, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08033533473887167, "packed_extra_cold_setup_ms": 1.6371579986298457, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.1894404093424482, "packed_extra_cold_setup_ms": 1.6371579986298457, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18880233377179412, "packed_extra_cold_setup_ms": 1.6371579986298457, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
