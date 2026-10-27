# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.949589 ms, packed 1.031168 ms (paired SIMT/packed 0.922, 95% bootstrap [0.917415910186763, 0.9262213181711506]); synchronized wall SIMT 0.955360 ms, packed 1.036887 ms (ratio 0.922).
- training: CUDA event SIMT 7.510016 ms, packed 7.717888 ms (paired SIMT/packed 0.976, 95% bootstrap [0.972604565743585, 0.9803686809169008]); synchronized wall SIMT 7.522295 ms, packed 7.729979 ms (ratio 0.976).
- Preparation: common CUDA/Torch initialization 212.630 ms; fixture owner setup 511.210 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.820 ms (p10-p90 467.519-492.596); packed median 468.498 ms (p10-p90 468.246-470.399); packed extra including CE-GEO host compile -1.180 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0815786520640055, "packed_extra_cold_setup_ms": -1.1804329988081008, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0815273330469305, "packed_extra_cold_setup_ms": -1.1804329988081008, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.2078717549641924, "packed_extra_cold_setup_ms": -1.1804329988081008, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.20768466735413948, "packed_extra_cold_setup_ms": -1.1804329988081008, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
