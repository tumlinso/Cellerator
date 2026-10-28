# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.063936 ms, packed 1.115136 ms (paired SIMT/packed 0.954, 95% bootstrap [0.9540862899421334, 0.9540862899421334]); synchronized wall SIMT 1.081405 ms, packed 1.137600 ms (ratio 0.951).
- training: CUDA event SIMT 2.992128 ms, packed 3.758080 ms (paired SIMT/packed 0.796, 95% bootstrap [0.7961852570621595, 0.7961852570621595]); synchronized wall SIMT 3.008967 ms, packed 3.774694 ms (ratio 0.797).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 0.985088 ms, packed-owner median 0.949248 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 209.148 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.066 ms; CE-GEO host compilation 0.077 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 41.346 ms (p10-p90 41.346-41.346); packed median 40.645 ms (p10-p90 40.645-40.645); packed extra including CE-GEO host compile -0.625 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.051200032234191895, "packed_extra_cold_setup_ms": -0.624533997324761, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.056195000070147216, "packed_extra_cold_setup_ms": -0.624533997324761, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.7659521102905273, "packed_extra_cold_setup_ms": -0.624533997324761, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.7657269998162519, "packed_extra_cold_setup_ms": -0.624533997324761, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
