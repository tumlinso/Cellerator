# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.962219 ms, packed 1.043456 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9187336718394989, 0.9248688035957362]); synchronized wall SIMT 0.967680 ms, packed 1.048800 ms (ratio 0.924).
- training: CUDA event SIMT 7.323989 ms, packed 7.460182 ms (paired SIMT/packed 0.981, 95% bootstrap [0.9767014112294916, 0.9883276750175602]); synchronized wall SIMT 7.337641 ms, packed 7.474115 ms (ratio 0.980).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.982123 ms, packed-owner median 1.967104 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.026 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.533 ms; CE-GEO host compilation 0.077 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.697 ms (p10-p90 40.340-41.189); packed median 40.530 ms (p10-p90 40.382-41.161); packed extra including CE-GEO host compile -0.089 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08123731613159191, "packed_extra_cold_setup_ms": -0.08935499499784783, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08112000068649661, "packed_extra_cold_setup_ms": -0.08935499499784783, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.13619232177734375, "packed_extra_cold_setup_ms": -0.08935499499784783, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.13647433418858146, "packed_extra_cold_setup_ms": -0.08935499499784783, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
