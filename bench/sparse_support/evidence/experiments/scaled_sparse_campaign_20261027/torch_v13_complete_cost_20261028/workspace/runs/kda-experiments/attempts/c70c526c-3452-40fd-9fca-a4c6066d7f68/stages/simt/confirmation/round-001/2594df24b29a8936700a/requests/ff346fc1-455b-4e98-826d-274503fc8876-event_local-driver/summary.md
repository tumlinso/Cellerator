# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.957440 ms, packed 1.027072 ms (paired SIMT/packed 0.927, 95% bootstrap [0.9196721286446585, 0.9344207392264287]); synchronized wall SIMT 0.963342 ms, packed 1.032696 ms (ratio 0.928).
- training: CUDA event SIMT 7.785472 ms, packed 7.987541 ms (paired SIMT/packed 0.974, 95% bootstrap [0.9730939438857118, 0.9756138843641167]); synchronized wall SIMT 7.800345 ms, packed 8.004509 ms (ratio 0.975).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.709397 ms, packed-owner median 1.738752 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 212.192 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 84.826 ms; CE-GEO host compilation 0.078 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 45.637 ms (p10-p90 45.238-46.122); packed median 45.511 ms (p10-p90 44.983-45.819); packed extra including CE-GEO host compile -0.048 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.06963205337524403, "packed_extra_cold_setup_ms": -0.047650999476900324, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.06935466687233816, "packed_extra_cold_setup_ms": -0.047650999476900324, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.20206896464029978, "packed_extra_cold_setup_ms": -0.047650999476900324, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.2041643335057115, "packed_extra_cold_setup_ms": -0.047650999476900324, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
