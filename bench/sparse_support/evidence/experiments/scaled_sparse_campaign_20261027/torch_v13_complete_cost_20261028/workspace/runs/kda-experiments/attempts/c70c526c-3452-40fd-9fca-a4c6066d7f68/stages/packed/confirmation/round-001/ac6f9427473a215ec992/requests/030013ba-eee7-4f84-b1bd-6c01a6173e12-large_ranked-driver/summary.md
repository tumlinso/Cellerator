# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.492395 ms, packed 8.002891 ms (paired SIMT/packed 0.687, 95% bootstrap [0.686680093058234, 0.6875944945352884]); synchronized wall SIMT 5.494684 ms, packed 8.005154 ms (ratio 0.687).
- training: CUDA event SIMT 13.106176 ms, packed 15.587680 ms (paired SIMT/packed 0.840, 95% bootstrap [0.8398472507501589, 0.8413261008352332]); synchronized wall SIMT 13.111551 ms, packed 15.593433 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.402304 ms, packed-owner median 2.398549 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.497 ms; fixture bundle load 26.511 ms; support-spec construction 93.780 ms; fixture owner setup 622.723 ms; CE-GEO host compilation 306.533 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 531.977 ms (p10-p90 526.747-554.774); packed median 662.554 ms (p10-p90 656.271-689.470); packed extra including CE-GEO host compile 437.109 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.510496139526368, "packed_extra_cold_setup_ms": 437.10944399936125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.510469999833731, "packed_extra_cold_setup_ms": 437.10944399936125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.4815038045247384, "packed_extra_cold_setup_ms": 437.10944399936125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.481882333086105, "packed_extra_cold_setup_ms": 437.10944399936125, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
