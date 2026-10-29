# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.488651 ms, packed 7.988565 ms (paired SIMT/packed 0.687, 95% bootstrap [0.6868350154683196, 0.6874411599169085]); synchronized wall SIMT 5.491040 ms, packed 7.990737 ms (ratio 0.687).
- training: CUDA event SIMT 13.092181 ms, packed 15.593814 ms (paired SIMT/packed 0.840, 95% bootstrap [0.8394500633481794, 0.8408771359129062]); synchronized wall SIMT 13.097487 ms, packed 15.598899 ms (ratio 0.840).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.380800 ms, packed-owner median 2.378411 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 225.743 ms; fixture bundle load 25.889 ms; support-spec construction 94.264 ms; fixture owner setup 630.007 ms; CE-GEO host compilation 345.903 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 528.013 ms (p10-p90 523.150-537.409); packed median 661.416 ms (p10-p90 653.062-701.109); packed extra including CE-GEO host compile 479.305 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.4999148050944013, "packed_extra_cold_setup_ms": 479.3053140019765, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.4996973321928326, "packed_extra_cold_setup_ms": 479.3053140019765, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.5016326904296875, "packed_extra_cold_setup_ms": 479.3053140019765, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.5014116666473747, "packed_extra_cold_setup_ms": 479.3053140019765, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
