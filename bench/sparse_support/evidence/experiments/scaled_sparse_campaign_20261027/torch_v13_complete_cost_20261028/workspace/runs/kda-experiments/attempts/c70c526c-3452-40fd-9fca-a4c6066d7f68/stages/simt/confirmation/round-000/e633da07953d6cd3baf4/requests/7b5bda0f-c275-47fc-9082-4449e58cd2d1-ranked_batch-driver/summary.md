# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.944811 ms, packed 1.026731 ms (paired SIMT/packed 0.920, 95% bootstrap [0.9096401462158348, 0.9262735188997783]); synchronized wall SIMT 0.950276 ms, packed 1.032562 ms (ratio 0.921).
- training: CUDA event SIMT 6.837931 ms, packed 6.882987 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9665660598978594, 0.9876767379136308]); synchronized wall SIMT 6.851371 ms, packed 6.896214 ms (ratio 0.973).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.669803 ms, packed-owner median 1.707008 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 215.023 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 87.276 ms; CE-GEO host compilation 0.083 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.664 ms (p10-p90 40.574-40.969); packed median 40.578 ms (p10-p90 40.293-40.717); packed extra including CE-GEO host compile -0.003 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08192006746927905, "packed_extra_cold_setup_ms": -0.002974000381072983, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08228566609128996, "packed_extra_cold_setup_ms": -0.002974000381072983, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.045056025187174775, "packed_extra_cold_setup_ms": -0.002974000381072983, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.044842333106013044, "packed_extra_cold_setup_ms": -0.002974000381072983, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
