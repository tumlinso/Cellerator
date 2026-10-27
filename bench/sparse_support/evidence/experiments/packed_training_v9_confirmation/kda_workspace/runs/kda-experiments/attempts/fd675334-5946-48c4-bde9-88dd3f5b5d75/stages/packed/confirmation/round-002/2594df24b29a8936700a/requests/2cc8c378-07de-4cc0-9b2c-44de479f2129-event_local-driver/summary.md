# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.936960 ms, packed 1.013760 ms (paired SIMT/packed 0.920, 95% bootstrap [0.9107443852055537, 0.9295631693289889]); synchronized wall SIMT 0.942098 ms, packed 1.021554 ms (ratio 0.920).
- training: CUDA event SIMT 7.312384 ms, packed 7.496363 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9719555871508341, 0.9835170068356074]); synchronized wall SIMT 7.333551 ms, packed 7.509279 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 212.630 ms; fixture owner setup 513.787 ms; CE-GEO host compilation 0.145 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.273 ms (p10-p90 468.143-489.962); packed median 467.604 ms (p10-p90 465.981-472.342); packed extra including CE-GEO host compile -1.524 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07680002848307299, "packed_extra_cold_setup_ms": -1.5244540045387112, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07945633357545978, "packed_extra_cold_setup_ms": -1.5244540045387112, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18397839864095022, "packed_extra_cold_setup_ms": -1.5244540045387112, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.17572833409455324, "packed_extra_cold_setup_ms": -1.5244540045387112, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
