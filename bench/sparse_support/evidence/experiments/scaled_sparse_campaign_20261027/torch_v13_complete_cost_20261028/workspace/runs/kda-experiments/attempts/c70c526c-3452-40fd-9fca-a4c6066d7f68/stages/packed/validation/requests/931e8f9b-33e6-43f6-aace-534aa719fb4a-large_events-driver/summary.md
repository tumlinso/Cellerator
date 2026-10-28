# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_events
- Shape: `{"active_entities_per_cell": [85, 73, 76, 77, 72, 80, 70, 81], "candidate_feature_workspace_estimate_bytes": 169900032, "candidate_items": 82959, "cells": 8, "dense_context_and_output_estimate_bytes": 245760000, "entities": 20000, "estimated_numerical_workspace_bytes_before_ce_geo": 1398700032, "event_occurrences_after_both_windows": [256, 256, 256, 256, 256, 256, 256, 256], "events_per_cell_in_full_declared_window": [256, 256, 256, 256, 256, 256, 256, 256], "global_edge_value_refresh_intermediate_estimate_bytes": 983040000, "max_candidate_items_per_cell": 13280, "relation_width": 64}`
- Work: 41481 primal / 82959 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.735424 ms, packed 6.437888 ms (paired SIMT/packed 0.891, 95% bootstrap [0.8908859415031075, 0.8908859415031075]); synchronized wall SIMT 5.740904 ms, packed 6.444766 ms (ratio 0.891).
- training: CUDA event SIMT 14.252032 ms, packed 14.846976 ms (paired SIMT/packed 0.960, 95% bootstrap [0.9599282716550801, 0.9599282716550801]); synchronized wall SIMT 14.267820 ms, packed 14.862599 ms (ratio 0.960).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 11.422720 ms, packed-owner median 11.431840 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.997 ms; fixture bundle load 130.300 ms; support-spec construction 593.015 ms; fixture owner setup 2612.253 ms; CE-GEO host compilation 1933.329 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 2386.110 ms (p10-p90 2386.110-2386.110); packed median 3152.294 ms (p10-p90 3152.294-3152.294); packed extra including CE-GEO host compile 2699.513 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.7024641036987305, "packed_extra_cold_setup_ms": 2699.513092000416, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.7038619987724815, "packed_extra_cold_setup_ms": 2699.513092000416, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.5949440002441406, "packed_extra_cold_setup_ms": 2699.513092000416, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.5947790014033671, "packed_extra_cold_setup_ms": 2699.513092000416, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 41478, 'packed': 41478}.
