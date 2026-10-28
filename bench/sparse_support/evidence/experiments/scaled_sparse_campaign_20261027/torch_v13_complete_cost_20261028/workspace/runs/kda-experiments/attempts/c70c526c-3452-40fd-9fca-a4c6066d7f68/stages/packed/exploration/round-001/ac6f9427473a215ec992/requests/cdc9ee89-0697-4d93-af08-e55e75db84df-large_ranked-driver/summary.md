# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.508096 ms, packed 8.000170 ms (paired SIMT/packed 0.688, 95% bootstrap [0.6873346568600528, 0.6886212032749137]); synchronized wall SIMT 5.510433 ms, packed 8.002346 ms (ratio 0.688).
- training: CUDA event SIMT 13.094229 ms, packed 15.582549 ms (paired SIMT/packed 0.841, 95% bootstrap [0.840166445251491, 0.8410919173242427]); synchronized wall SIMT 13.099795 ms, packed 15.587883 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.380800 ms, packed-owner median 2.377387 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 217.001 ms; fixture bundle load 26.388 ms; support-spec construction 93.597 ms; fixture owner setup 614.811 ms; CE-GEO host compilation 313.466 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 534.824 ms (p10-p90 527.877-549.985); packed median 666.248 ms (p10-p90 646.076-678.252); packed extra including CE-GEO host compile 444.891 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.492074330647786, "packed_extra_cold_setup_ms": 444.89066900496255, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.4919129997821683, "packed_extra_cold_setup_ms": 444.89066900496255, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.4883193969726562, "packed_extra_cold_setup_ms": 444.89066900496255, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.4880879997605607, "packed_extra_cold_setup_ms": 444.89066900496255, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
