# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.488619 ms, packed 7.989930 ms (paired SIMT/packed 0.687, 95% bootstrap [0.6864127853068452, 0.687349298992151]); synchronized wall SIMT 5.490812 ms, packed 7.992005 ms (ratio 0.687).
- training: CUDA event SIMT 13.115392 ms, packed 15.597226 ms (paired SIMT/packed 0.841, 95% bootstrap [0.8401172566387914, 0.8409717683493988]); synchronized wall SIMT 13.120460 ms, packed 15.602624 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.399915 ms, packed-owner median 2.400597 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 212.702 ms; fixture bundle load 26.441 ms; support-spec construction 93.749 ms; fixture owner setup 631.968 ms; CE-GEO host compilation 309.928 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 529.026 ms (p10-p90 523.579-579.493); packed median 659.852 ms (p10-p90 656.290-689.788); packed extra including CE-GEO host compile 440.754 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.501311620076497, "packed_extra_cold_setup_ms": 440.75388699639007, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.501193666830659, "packed_extra_cold_setup_ms": 440.75388699639007, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.4818344116210938, "packed_extra_cold_setup_ms": 440.75388699639007, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.482163332994485, "packed_extra_cold_setup_ms": 440.75388699639007, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
