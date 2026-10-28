# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.939200 ms, packed 8.561664 ms (paired SIMT/packed 0.694, 95% bootstrap [0.693696947554206, 0.693696947554206]); synchronized wall SIMT 5.945037 ms, packed 8.570897 ms (ratio 0.694).
- training: CUDA event SIMT 13.863936 ms, packed 16.391169 ms (paired SIMT/packed 0.846, 95% bootstrap [0.8458174500764689, 0.8458174500764689]); synchronized wall SIMT 13.881827 ms, packed 16.415360 ms (ratio 0.846).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.584576 ms, packed-owner median 2.580480 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 213.561 ms; fixture bundle load 26.490 ms; support-spec construction 94.445 ms; fixture owner setup 639.966 ms; CE-GEO host compilation 310.794 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 556.255 ms (p10-p90 556.255-556.255); packed median 657.959 ms (p10-p90 657.959-657.959); packed extra including CE-GEO host compile 412.498 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.6224637031555176, "packed_extra_cold_setup_ms": 412.49829399748705, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.6258600009896327, "packed_extra_cold_setup_ms": 412.49829399748705, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.5272321701049805, "packed_extra_cold_setup_ms": 412.49829399748705, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.533532999223098, "packed_extra_cold_setup_ms": 412.49829399748705, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
