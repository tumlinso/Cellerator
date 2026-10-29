# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.063936 ms, packed 1.104896 ms (paired SIMT/packed 0.963, 95% bootstrap [0.9629286773244413, 0.9629286773244413]); synchronized wall SIMT 1.079981 ms, packed 1.122195 ms (ratio 0.962).
- training: CUDA event SIMT 3.663872 ms, packed 3.782656 ms (paired SIMT/packed 0.969, 95% bootstrap [0.9685977386685881, 0.9685977386685881]); synchronized wall SIMT 3.680287 ms, packed 3.798388 ms (ratio 0.969).
- Preparation: common CUDA/Torch initialization 205.479 ms; fixture owner setup 535.649 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 469.004 ms (p10-p90 469.004-469.004); packed median 486.058 ms (p10-p90 486.058-486.058); packed extra including CE-GEO host compile 17.196 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.040959954261779785, "packed_extra_cold_setup_ms": 17.19647299978533, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.042213996493956074, "packed_extra_cold_setup_ms": 17.19647299978533, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.11878395080566406, "packed_extra_cold_setup_ms": 17.19647299978533, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1181009974970948, "packed_extra_cold_setup_ms": 17.19647299978533, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
