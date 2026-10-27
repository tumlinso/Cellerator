# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.962560 ms, packed 1.042773 ms (paired SIMT/packed 0.923, 95% bootstrap [0.917741912346175, 0.9273265224183517]); synchronized wall SIMT 0.968192 ms, packed 1.048180 ms (ratio 0.923).
- training: CUDA event SIMT 5.263701 ms, packed 5.563733 ms (paired SIMT/packed 0.977, 95% bootstrap [0.9555254608555499, 0.9920774083932831]); synchronized wall SIMT 5.272327 ms, packed 5.573719 ms (ratio 0.977).
- Preparation: common CUDA/Torch initialization 217.341 ms; fixture owner setup 504.055 ms; CE-GEO host compilation 0.154 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 476.712 ms (p10-p90 464.037-510.377); packed median 498.859 ms (p10-p90 489.445-512.976); packed extra including CE-GEO host compile 22.301 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08021330833435047, "packed_extra_cold_setup_ms": 22.30106400384102, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0799883321936552, "packed_extra_cold_setup_ms": 22.30106400384102, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.3000319798787432, "packed_extra_cold_setup_ms": 22.30106400384102, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.30139200074093697, "packed_extra_cold_setup_ms": 22.30106400384102, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
