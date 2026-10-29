# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.897365 ms, packed 0.975872 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9162321964880205, 0.9278169249351538]); synchronized wall SIMT 0.902563 ms, packed 0.981166 ms (ratio 0.924).
- training: CUDA event SIMT 6.567253 ms, packed 6.751232 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9700611314748949, 0.9750316871367393]); synchronized wall SIMT 6.580694 ms, packed 6.766237 ms (ratio 0.973).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.779712 ms, packed-owner median 1.761621 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 211.956 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.386 ms; CE-GEO host compilation 0.080 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.883 ms (p10-p90 40.606-41.530); packed median 41.120 ms (p10-p90 41.059-41.305); packed extra including CE-GEO host compile 0.318 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07850662867228186, "packed_extra_cold_setup_ms": 0.3177170001436025, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07860333304658229, "packed_extra_cold_setup_ms": 0.3177170001436025, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18397903442382812, "packed_extra_cold_setup_ms": 0.3177170001436025, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18554333413097446, "packed_extra_cold_setup_ms": 0.3177170001436025, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
