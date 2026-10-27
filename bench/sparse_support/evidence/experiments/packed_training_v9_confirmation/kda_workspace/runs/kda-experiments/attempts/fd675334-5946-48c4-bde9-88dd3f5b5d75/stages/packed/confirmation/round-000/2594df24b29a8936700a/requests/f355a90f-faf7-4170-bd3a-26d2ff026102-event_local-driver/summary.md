# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.216512 ms, packed 1.305941 ms (paired SIMT/packed 0.933, 95% bootstrap [0.9287211795044357, 0.9383687566061804]); synchronized wall SIMT 1.222622 ms, packed 1.311688 ms (ratio 0.934).
- training: CUDA event SIMT 4.706304 ms, packed 4.807339 ms (paired SIMT/packed 0.979, 95% bootstrap [0.9716720020868646, 0.9896826240420379]); synchronized wall SIMT 4.712035 ms, packed 4.813137 ms (ratio 0.979).
- Preparation: common CUDA/Torch initialization 209.988 ms; fixture owner setup 510.420 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 459.673 ms (p10-p90 454.220-464.086); packed median 459.993 ms (p10-p90 458.465-462.280); packed extra including CE-GEO host compile 0.461 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08942937850952148, "packed_extra_cold_setup_ms": 0.4614569988916628, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08906666698749177, "packed_extra_cold_setup_ms": 0.4614569988916628, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.10103480021158884, "packed_extra_cold_setup_ms": 0.4614569988916628, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.10110199946211651, "packed_extra_cold_setup_ms": 0.4614569988916628, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
