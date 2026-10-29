# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_events
- Shape: `{"active_entities_per_cell": [85, 73, 76, 77, 72, 80, 70, 81], "candidate_feature_workspace_estimate_bytes": 169900032, "candidate_items": 82959, "cells": 8, "dense_context_and_output_estimate_bytes": 245760000, "entities": 20000, "estimated_numerical_workspace_bytes_before_ce_geo": 1398700032, "event_occurrences_after_both_windows": [256, 256, 256, 256, 256, 256, 256, 256], "events_per_cell_in_full_declared_window": [256, 256, 256, 256, 256, 256, 256, 256], "global_edge_value_refresh_intermediate_estimate_bytes": 983040000, "max_candidate_items_per_cell": 13280, "relation_width": 64}`
- Work: 41481 primal / 82959 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.538474 ms, packed 6.251862 ms (paired SIMT/packed 0.887, 95% bootstrap [0.885262116314054, 0.8878121363614576]); synchronized wall SIMT 5.540702 ms, packed 6.254142 ms (ratio 0.887).
- training: CUDA event SIMT 14.050987 ms, packed 14.749013 ms (paired SIMT/packed 0.953, 95% bootstrap [0.9520746771890554, 0.9531550237505052]); synchronized wall SIMT 14.056349 ms, packed 14.753653 ms (ratio 0.953).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 11.304619 ms, packed-owner median 11.307691 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 213.145 ms; fixture bundle load 130.817 ms; support-spec construction 557.229 ms; fixture owner setup 2651.746 ms; CE-GEO host compilation 2064.861 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 2527.748 ms (p10-p90 2493.657-2627.406); packed median 3205.129 ms (p10-p90 3173.271-3221.087); packed extra including CE-GEO host compile 2742.242 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.7133871714274092, "packed_extra_cold_setup_ms": 2742.2419050017197, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.7134406672169762, "packed_extra_cold_setup_ms": 2742.2419050017197, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.6980260213216152, "packed_extra_cold_setup_ms": 2742.2419050017197, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.6973039999138564, "packed_extra_cold_setup_ms": 2742.2419050017197, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 41478, 'packed': 41478}.
