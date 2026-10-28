# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.960853 ms, packed 1.037312 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9177131414124542, 0.9274834298976602]); synchronized wall SIMT 0.966412 ms, packed 1.042688 ms (ratio 0.923).
- training: CUDA event SIMT 6.863530 ms, packed 7.244459 ms (paired SIMT/packed 0.971, 95% bootstrap [0.9664079803445288, 0.9860047720942872]); synchronized wall SIMT 6.876036 ms, packed 7.258588 ms (ratio 0.971).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.784832 ms, packed-owner median 1.831253 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 230.806 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.997 ms; CE-GEO host compilation 0.079 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.583 ms (p10-p90 40.043-40.839); packed median 40.472 ms (p10-p90 40.297-40.667); packed extra including CE-GEO host compile -0.033 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07645869255065918, "packed_extra_cold_setup_ms": -0.03293100235168822, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07627533462558256, "packed_extra_cold_setup_ms": -0.03293100235168822, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.38092803955078125, "packed_extra_cold_setup_ms": -0.03293100235168822, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.38255200100441744, "packed_extra_cold_setup_ms": -0.03293100235168822, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
