# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.485909 ms, packed 7.991979 ms (paired SIMT/packed 0.687, 95% bootstrap [0.6864710356922259, 0.687564005574263]); synchronized wall SIMT 5.488341 ms, packed 7.994209 ms (ratio 0.687).
- training: CUDA event SIMT 13.123243 ms, packed 15.609856 ms (paired SIMT/packed 0.840, 95% bootstrap [0.8398111279553299, 0.8407478686472993]); synchronized wall SIMT 13.128456 ms, packed 15.615102 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.405376 ms, packed-owner median 2.405035 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 214.073 ms; fixture bundle load 26.570 ms; support-spec construction 98.675 ms; fixture owner setup 659.726 ms; CE-GEO host compilation 309.612 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 579.281 ms (p10-p90 528.510-589.665); packed median 661.967 ms (p10-p90 653.978-717.954); packed extra including CE-GEO host compile 392.297 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.5060698191324864, "packed_extra_cold_setup_ms": 392.29687999613816, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.5058686660486273, "packed_extra_cold_setup_ms": 392.29687999613816, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.486612955729166, "packed_extra_cold_setup_ms": 392.29687999613816, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.4866463342429306, "packed_extra_cold_setup_ms": 392.29687999613816, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
