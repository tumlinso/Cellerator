# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.953344 ms, packed 1.036288 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9169960307591053, 0.9256578671748705]); synchronized wall SIMT 0.958732 ms, packed 1.041642 ms (ratio 0.924).
- training: CUDA event SIMT 7.527082 ms, packed 7.720619 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9707702464659201, 0.9756915235470904]); synchronized wall SIMT 7.539964 ms, packed 7.732737 ms (ratio 0.973).
- Preparation: common CUDA/Torch initialization 211.860 ms; fixture owner setup 521.088 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 466.835 ms (p10-p90 465.620-488.100); packed median 469.117 ms (p10-p90 468.170-472.980); packed extra including CE-GEO host compile 2.424 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08294399579366052, "packed_extra_cold_setup_ms": 2.423568002996035, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08291000024958828, "packed_extra_cold_setup_ms": 2.423568002996035, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19353612263997366, "packed_extra_cold_setup_ms": 2.423568002996035, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1927726674087653, "packed_extra_cold_setup_ms": 2.423568002996035, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
