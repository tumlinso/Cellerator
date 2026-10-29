# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.927403 ms, packed 1.008640 ms (paired SIMT/packed 0.920, 95% bootstrap [0.9179714190480823, 0.9227380947872443]); synchronized wall SIMT 0.932880 ms, packed 1.013994 ms (ratio 0.921).
- training: CUDA event SIMT 6.901760 ms, packed 7.069354 ms (paired SIMT/packed 0.973, 95% bootstrap [0.955823436640108, 0.9870715420785218]); synchronized wall SIMT 6.913993 ms, packed 7.080754 ms (ratio 0.973).
- Preparation: common CUDA/Torch initialization 214.835 ms; fixture owner setup 512.265 ms; CE-GEO host compilation 0.140 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.759 ms (p10-p90 464.034-492.507); packed median 466.370 ms (p10-p90 463.610-468.863); packed extra including CE-GEO host compile -3.248 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08123731613159191, "packed_extra_cold_setup_ms": -3.2481890011695214, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08111399923412432, "packed_extra_cold_setup_ms": -3.2481890011695214, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.16759427388509085, "packed_extra_cold_setup_ms": -3.2481890011695214, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.16676033313463723, "packed_extra_cold_setup_ms": -3.2481890011695214, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
