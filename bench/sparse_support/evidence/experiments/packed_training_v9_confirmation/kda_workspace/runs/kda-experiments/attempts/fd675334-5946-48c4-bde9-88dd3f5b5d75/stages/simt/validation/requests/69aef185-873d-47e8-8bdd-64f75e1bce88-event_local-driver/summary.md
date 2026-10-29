# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.986112 ms, packed 1.048576 ms (paired SIMT/packed 0.940, 95% bootstrap [0.9404296884992007, 0.9404296884992007]); synchronized wall SIMT 1.000756 ms, packed 1.064652 ms (ratio 0.940).
- training: CUDA event SIMT 3.357696 ms, packed 3.444736 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9747324766217685, 0.9747324766217685]); synchronized wall SIMT 3.373785 ms, packed 3.459951 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 213.982 ms; fixture owner setup 509.874 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 481.059 ms (p10-p90 481.059-481.059); packed median 468.737 ms (p10-p90 468.737-468.737); packed extra including CE-GEO host compile -12.181 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.062463998794555664, "packed_extra_cold_setup_ms": -12.18133599832072, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0638960009382572, "packed_extra_cold_setup_ms": -12.18133599832072, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08703994750976562, "packed_extra_cold_setup_ms": -12.18133599832072, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08616599734523334, "packed_extra_cold_setup_ms": -12.18133599832072, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
