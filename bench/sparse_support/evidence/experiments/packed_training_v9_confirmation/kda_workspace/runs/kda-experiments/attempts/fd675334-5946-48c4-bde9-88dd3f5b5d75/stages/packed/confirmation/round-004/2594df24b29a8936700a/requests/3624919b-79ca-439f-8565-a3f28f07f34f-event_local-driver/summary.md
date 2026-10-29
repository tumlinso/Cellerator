# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.924672 ms, packed 1.005909 ms (paired SIMT/packed 0.917, 95% bootstrap [0.9121139797218498, 0.9242579108998186]); synchronized wall SIMT 0.930263 ms, packed 1.010988 ms (ratio 0.918).
- training: CUDA event SIMT 6.983680 ms, packed 7.146496 ms (paired SIMT/packed 0.982, 95% bootstrap [0.9728398769152706, 0.9916482760483151]); synchronized wall SIMT 6.995897 ms, packed 7.159592 ms (ratio 0.982).
- Preparation: common CUDA/Torch initialization 214.311 ms; fixture owner setup 511.229 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.223 ms (p10-p90 466.502-481.290); packed median 468.243 ms (p10-p90 466.104-470.955); packed extra including CE-GEO host compile 0.161 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08123739560445153, "packed_extra_cold_setup_ms": 0.16133799726958387, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0807249998615589, "packed_extra_cold_setup_ms": 0.16133799726958387, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.16281572977701853, "packed_extra_cold_setup_ms": 0.16133799726958387, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.16369466781422215, "packed_extra_cold_setup_ms": 0.16133799726958387, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
