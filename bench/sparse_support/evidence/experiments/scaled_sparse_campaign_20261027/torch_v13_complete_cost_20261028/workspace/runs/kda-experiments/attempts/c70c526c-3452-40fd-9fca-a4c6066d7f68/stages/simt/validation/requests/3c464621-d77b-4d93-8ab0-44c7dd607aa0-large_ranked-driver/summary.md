# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### large_ranked
- Shape: `{"active_entities_per_cell": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], "candidate_feature_workspace_estimate_bytes": 2129813504, "candidate_items": 1039948, "cells": 32, "dense_context_and_output_estimate_bytes": 201326592, "entities": 4096, "estimated_numerical_workspace_bytes_before_ce_geo": 2532466688, "global_edge_value_refresh_intermediate_estimate_bytes": 201326592, "max_candidate_items_per_cell": 35268, "relation_width": 64, "selected_genes_per_cell": 512}`
- Work: 519984 primal / 1039948 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 6.068224 ms, packed 8.755200 ms (paired SIMT/packed 0.693, 95% bootstrap [0.6930993793034862, 0.6930993793034862]); synchronized wall SIMT 6.073349 ms, packed 8.762509 ms (ratio 0.693).
- training: CUDA event SIMT 14.071808 ms, packed 16.768000 ms (paired SIMT/packed 0.839, 95% bootstrap [0.8392061161646777, 0.8392061161646777]); synchronized wall SIMT 14.088660 ms, packed 16.785129 ms (ratio 0.839).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.621440 ms, packed-owner median 2.623488 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 245.648 ms; fixture bundle load 29.154 ms; support-spec construction 100.238 ms; fixture owner setup 713.935 ms; CE-GEO host compilation 326.691 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 537.292 ms (p10-p90 537.292-537.292); packed median 661.486 ms (p10-p90 661.486-661.486); packed extra including CE-GEO host compile 450.884 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.686976432800293, "packed_extra_cold_setup_ms": 450.88408299852745, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.6891599991358817, "packed_extra_cold_setup_ms": 450.88408299852745, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -2.6961917877197266, "packed_extra_cold_setup_ms": 450.88408299852745, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -2.69646900051157, "packed_extra_cold_setup_ms": 450.88408299852745, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 519961, 'packed': 519961}.
