# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.967339 ms, packed 1.051307 ms (paired SIMT/packed 0.922, 95% bootstrap [0.9198174692797124, 0.9236541116637516]); synchronized wall SIMT 0.971915 ms, packed 1.057007 ms (ratio 0.922).
- training: CUDA event SIMT 7.602176 ms, packed 7.796395 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9730556206063753, 0.9774995334010259]); synchronized wall SIMT 7.614669 ms, packed 7.809340 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 212.781 ms; fixture owner setup 511.195 ms; CE-GEO host compilation 0.140 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.522 ms (p10-p90 465.479-495.034); packed median 468.470 ms (p10-p90 464.181-472.117); packed extra including CE-GEO host compile 0.089 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08396800359090173, "packed_extra_cold_setup_ms": 0.0887890018930193, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0850919993051017, "packed_extra_cold_setup_ms": 0.0887890018930193, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19421895345052143, "packed_extra_cold_setup_ms": 0.0887890018930193, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.19467133339882547, "packed_extra_cold_setup_ms": 0.0887890018930193, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
