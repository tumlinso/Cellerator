# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.949248 ms, packed 1.030144 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9168323984233863, 0.9247999500513084]); synchronized wall SIMT 0.954964 ms, packed 1.035219 ms (ratio 0.923).
- training: CUDA event SIMT 7.454379 ms, packed 7.628800 ms (paired SIMT/packed 0.976, 95% bootstrap [0.9741513536600352, 0.9777373791773809]); synchronized wall SIMT 7.467640 ms, packed 7.641096 ms (ratio 0.976).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.207744 ms, packed-owner median 2.239147 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 223.686 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 78.066 ms; CE-GEO host compilation 0.074 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.494 ms (p10-p90 40.030-40.681); packed median 40.244 ms (p10-p90 39.987-40.803); packed extra including CE-GEO host compile -0.175 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0808959801991781, "packed_extra_cold_setup_ms": -0.1754609984345734, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08025533437224419, "packed_extra_cold_setup_ms": -0.1754609984345734, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.1744213104248047, "packed_extra_cold_setup_ms": -0.1754609984345734, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1734563338686712, "packed_extra_cold_setup_ms": -0.1754609984345734, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
