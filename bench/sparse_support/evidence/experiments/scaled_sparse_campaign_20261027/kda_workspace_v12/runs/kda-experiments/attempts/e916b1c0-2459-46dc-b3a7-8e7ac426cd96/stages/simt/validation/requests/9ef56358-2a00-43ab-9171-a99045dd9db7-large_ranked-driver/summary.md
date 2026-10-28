# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 6.190080 ms, packed 8.860672 ms (paired SIMT/packed 0.699, 95% bootstrap [0.6986016599993456, 0.6986016599993456]); synchronized wall SIMT 6.194087 ms, packed 8.868017 ms (ratio 0.698).
- training: CUDA event SIMT 14.291968 ms, packed 17.020927 ms (paired SIMT/packed 0.840, 95% bootstrap [0.8396703649135105, 0.8396703649135105]); synchronized wall SIMT 14.309674 ms, packed 17.039016 ms (ratio 0.840).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.673664 ms, packed-owner median 2.667520 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 1531.482 ms; fixture bundle load 26.774 ms; support-spec construction 99.516 ms; fixture owner setup 38572.412 ms; CE-GEO host compilation 320.767 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 37930.515 ms (p10-p90 37930.515-37930.515); packed median 38469.494 ms (p10-p90 38469.494-38469.494); packed extra including CE-GEO host compile 859.746 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.6705918312072754, "packed_extra_cold_setup_ms": 859.7458059994096, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.673930001037661, "packed_extra_cold_setup_ms": 859.7458059994096, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.728959083557129, "packed_extra_cold_setup_ms": 859.7458059994096, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.729341998929158, "packed_extra_cold_setup_ms": 859.7458059994096, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
