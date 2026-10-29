# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.973141 ms, packed 1.053355 ms (paired SIMT/packed 0.923, 95% bootstrap [0.915045427267262, 0.9264470364650604]); synchronized wall SIMT 0.979152 ms, packed 1.059069 ms (ratio 0.923).
- training: CUDA event SIMT 5.942954 ms, packed 6.036139 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9527767355244156, 0.9965324381524728]); synchronized wall SIMT 5.954340 ms, packed 6.046717 ms (ratio 0.973).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.620309 ms, packed-owner median 1.640107 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 209.364 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.406 ms; CE-GEO host compilation 0.078 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 41.067 ms (p10-p90 40.459-43.852); packed median 41.207 ms (p10-p90 40.891-47.555); packed extra including CE-GEO host compile 0.218 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0802133083343507, "packed_extra_cold_setup_ms": 0.21787899822811596, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07991766688064683, "packed_extra_cold_setup_ms": 0.21787899822811596, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.09318415323893259, "packed_extra_cold_setup_ms": 0.21787899822811596, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.09237633397181799, "packed_extra_cold_setup_ms": 0.21787899822811596, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
