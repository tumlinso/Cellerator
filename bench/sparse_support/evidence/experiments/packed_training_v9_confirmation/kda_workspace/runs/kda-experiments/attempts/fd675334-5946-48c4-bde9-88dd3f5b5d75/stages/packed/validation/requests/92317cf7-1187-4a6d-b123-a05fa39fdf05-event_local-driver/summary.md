# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.990208 ms, packed 1.067008 ms (paired SIMT/packed 0.928, 95% bootstrap [0.9280230444237285, 0.9280230444237285]); synchronized wall SIMT 1.004661 ms, packed 1.083721 ms (ratio 0.927).
- training: CUDA event SIMT 3.508224 ms, packed 3.434496 ms (paired SIMT/packed 1.021, 95% bootstrap [1.0214669302729824, 1.0214669302729824]); synchronized wall SIMT 3.523582 ms, packed 3.450662 ms (ratio 1.021).
- Preparation: common CUDA/Torch initialization 221.376 ms; fixture owner setup 515.467 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 469.669 ms (p10-p90 469.669-469.669); packed median 471.016 ms (p10-p90 471.016-471.016); packed extra including CE-GEO host compile 1.488 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07679998874664307, "packed_extra_cold_setup_ms": 1.487788002123125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07905999882495962, "packed_extra_cold_setup_ms": 1.487788002123125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": 20.179393116166292, "median_step_saving_ms": 0.07372808456420898, "packed_extra_cold_setup_ms": 1.487788002123125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": true}, "training_synchronized_wall": {"break_even_steps": 20.403017007950023, "median_step_saving_ms": 0.07292000009329058, "packed_extra_cold_setup_ms": 1.487788002123125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": true}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
