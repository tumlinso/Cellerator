# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.937643 ms, packed 1.015467 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9177049005126091, 0.9291945630452623]); synchronized wall SIMT 0.942625 ms, packed 1.020965 ms (ratio 0.924).
- training: CUDA event SIMT 5.654528 ms, packed 5.959339 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9698937427291965, 0.9827042747557024]); synchronized wall SIMT 5.663887 ms, packed 5.968848 ms (ratio 0.973).
- Preparation: common CUDA/Torch initialization 225.264 ms; fixture owner setup 516.408 ms; CE-GEO host compilation 0.143 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.795 ms (p10-p90 467.564-490.192); packed median 467.576 ms (p10-p90 462.447-468.342); packed extra including CE-GEO host compile -2.076 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07782403628031409, "packed_extra_cold_setup_ms": -2.07643500471022, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0783396668945594, "packed_extra_cold_setup_ms": -2.07643500471022, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.304810206095377, "packed_extra_cold_setup_ms": -2.07643500471022, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.3049610010445267, "packed_extra_cold_setup_ms": -2.07643500471022, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
