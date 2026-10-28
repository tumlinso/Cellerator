# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.489664 ms, packed 7.993003 ms (paired SIMT/packed 0.687, 95% bootstrap [0.6867418683511199, 0.6873690843898872]); synchronized wall SIMT 5.492937 ms, packed 7.995642 ms (ratio 0.687).
- training: CUDA event SIMT 13.113344 ms, packed 15.595520 ms (paired SIMT/packed 0.841, 95% bootstrap [0.8400428953549062, 0.8413164891239854]); synchronized wall SIMT 13.118877 ms, packed 15.601069 ms (ratio 0.841).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.404011 ms, packed-owner median 2.403328 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 207.273 ms; fixture bundle load 26.478 ms; support-spec construction 94.061 ms; fixture owner setup 620.626 ms; CE-GEO host compilation 309.187 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 539.974 ms (p10-p90 523.998-552.459); packed median 664.647 ms (p10-p90 649.411-689.706); packed extra including CE-GEO host compile 433.860 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.5033384958902998, "packed_extra_cold_setup_ms": 433.8601119998202, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.5027056666052276, "packed_extra_cold_setup_ms": 433.8601119998202, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.482176462809244, "packed_extra_cold_setup_ms": 433.8601119998202, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.4821916679987535, "packed_extra_cold_setup_ms": 433.8601119998202, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
