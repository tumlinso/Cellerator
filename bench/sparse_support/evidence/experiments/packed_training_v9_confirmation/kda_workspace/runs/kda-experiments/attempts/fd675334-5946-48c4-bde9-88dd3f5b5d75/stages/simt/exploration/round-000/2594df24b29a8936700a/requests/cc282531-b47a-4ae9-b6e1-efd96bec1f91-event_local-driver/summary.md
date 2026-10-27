# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.019563 ms, packed 1.093632 ms (paired SIMT/packed 0.921, 95% bootstrap [0.9091739813287, 0.940715169836367]); synchronized wall SIMT 1.024763 ms, packed 1.099124 ms (ratio 0.918).
- training: CUDA event SIMT 6.914389 ms, packed 6.964565 ms (paired SIMT/packed 0.979, 95% bootstrap [0.9503636287212817, 0.9946897257510812]); synchronized wall SIMT 6.926703 ms, packed 6.976148 ms (ratio 0.979).
- Preparation: common CUDA/Torch initialization 234.715 ms; fixture owner setup 513.103 ms; CE-GEO host compilation 0.169 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 472.796 ms (p10-p90 468.566-496.431); packed median 470.392 ms (p10-p90 465.127-475.825); packed extra including CE-GEO host compile -2.234 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07406934102376295, "packed_extra_cold_setup_ms": -2.2344620010699145, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07436166682358203, "packed_extra_cold_setup_ms": -2.2344620010699145, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.05017598470052054, "packed_extra_cold_setup_ms": -2.2344620010699145, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.04944466733528863, "packed_extra_cold_setup_ms": -2.2344620010699145, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
