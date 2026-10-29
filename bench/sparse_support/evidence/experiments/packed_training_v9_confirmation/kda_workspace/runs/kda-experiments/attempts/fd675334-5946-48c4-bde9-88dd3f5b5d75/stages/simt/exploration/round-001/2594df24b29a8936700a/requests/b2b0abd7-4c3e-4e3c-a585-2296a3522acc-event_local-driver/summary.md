# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.943787 ms, packed 1.028096 ms (paired SIMT/packed 0.918, 95% bootstrap [0.9167221727054237, 0.9202921989674259]); synchronized wall SIMT 0.949015 ms, packed 1.032894 ms (ratio 0.919).
- training: CUDA event SIMT 7.428779 ms, packed 7.708331 ms (paired SIMT/packed 0.979, 95% bootstrap [0.9711064386370235, 0.9822446088099556]); synchronized wall SIMT 7.441295 ms, packed 7.720895 ms (ratio 0.980).
- Preparation: common CUDA/Torch initialization 215.179 ms; fixture owner setup 510.356 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.177 ms (p10-p90 464.556-479.349); packed median 476.222 ms (p10-p90 467.789-491.445); packed extra including CE-GEO host compile 7.186 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08430926005045569, "packed_extra_cold_setup_ms": 7.18612500349991, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08387900015804917, "packed_extra_cold_setup_ms": 7.18612500349991, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.27955182393391986, "packed_extra_cold_setup_ms": 7.18612500349991, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.2795999995820848, "packed_extra_cold_setup_ms": 7.18612500349991, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
