# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.499904 ms, packed 8.015531 ms (paired SIMT/packed 0.687, 95% bootstrap [0.686134184238809, 0.688033213455684]); synchronized wall SIMT 5.502205 ms, packed 8.017858 ms (ratio 0.687).
- training: CUDA event SIMT 13.116416 ms, packed 15.592448 ms (paired SIMT/packed 0.841, 95% bootstrap [0.8404689294948804, 0.8413872719400678]); synchronized wall SIMT 13.121795 ms, packed 15.598324 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.403669 ms, packed-owner median 2.401280 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.939 ms; fixture bundle load 26.529 ms; support-spec construction 93.688 ms; fixture owner setup 618.294 ms; CE-GEO host compilation 310.401 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 530.095 ms (p10-p90 525.662-572.114); packed median 665.848 ms (p10-p90 655.336-703.181); packed extra including CE-GEO host compile 446.155 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.5156269073486337, "packed_extra_cold_setup_ms": 446.15512900054455, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.515652999136365, "packed_extra_cold_setup_ms": 446.15512900054455, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.476032257080078, "packed_extra_cold_setup_ms": 446.15512900054455, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.476529333459135, "packed_extra_cold_setup_ms": 446.15512900054455, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
