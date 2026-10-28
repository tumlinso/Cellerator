# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_events
- Shape: `{"active_entities_per_cell": [85, 73, 76, 77, 72, 80, 70, 81], "candidate_feature_workspace_estimate_bytes": 169900032, "candidate_items": 82959, "cells": 8, "dense_context_and_output_estimate_bytes": 245760000, "entities": 20000, "estimated_numerical_workspace_bytes_before_ce_geo": 1398700032, "event_occurrences_after_both_windows": [256, 256, 256, 256, 256, 256, 256, 256], "events_per_cell_in_full_declared_window": [256, 256, 256, 256, 256, 256, 256, 256], "global_edge_value_refresh_intermediate_estimate_bytes": 983040000, "max_candidate_items_per_cell": 13280, "relation_width": 64}`
- Work: 41481 primal / 82959 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 5.536768 ms, packed 6.252202 ms (paired SIMT/packed 0.886, 95% bootstrap [0.8842983582739848, 0.8864926503010985]); synchronized wall SIMT 5.539988 ms, packed 6.254927 ms (ratio 0.886).
- training: CUDA event SIMT 14.074197 ms, packed 14.769493 ms (paired SIMT/packed 0.953, 95% bootstrap [0.9525040173416505, 0.9539235933297712]); synchronized wall SIMT 14.079129 ms, packed 14.774647 ms (ratio 0.953).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 11.304960 ms, packed-owner median 11.304619 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 214.618 ms; fixture bundle load 131.510 ms; support-spec construction 561.103 ms; fixture owner setup 2753.076 ms; CE-GEO host compilation 1986.277 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 2535.758 ms (p10-p90 2487.526-2754.185); packed median 3253.727 ms (p10-p90 3194.164-3366.849); packed extra including CE-GEO host compile 2704.246 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.715434392293294, "packed_extra_cold_setup_ms": 2704.2460840020794, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.7149393325865576, "packed_extra_cold_setup_ms": 2704.2460840020794, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.6952959696451817, "packed_extra_cold_setup_ms": 2704.2460840020794, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.6955186666649151, "packed_extra_cold_setup_ms": 2704.2460840020794, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 41478, 'packed': 41478}.
