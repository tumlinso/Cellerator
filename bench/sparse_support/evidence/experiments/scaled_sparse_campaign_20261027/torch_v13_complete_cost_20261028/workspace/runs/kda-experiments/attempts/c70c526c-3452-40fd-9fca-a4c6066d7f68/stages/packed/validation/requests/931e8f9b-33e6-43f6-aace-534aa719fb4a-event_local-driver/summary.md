# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.992256 ms, packed 1.043456 ms (paired SIMT/packed 0.951, 95% bootstrap [0.9509323108282363, 0.9509323108282363]); synchronized wall SIMT 1.008050 ms, packed 1.060263 ms (ratio 0.951).
- training: CUDA event SIMT 3.470336 ms, packed 3.500032 ms (paired SIMT/packed 0.992, 95% bootstrap [0.9915155095319559, 0.9915155095319559]); synchronized wall SIMT 3.486479 ms, packed 3.517273 ms (ratio 0.991).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.078272 ms, packed-owner median 0.915456 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 209.663 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 97.404 ms; CE-GEO host compilation 0.112 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 75.693 ms (p10-p90 75.693-75.693); packed median 60.363 ms (p10-p90 60.363-60.363); packed extra including CE-GEO host compile -15.217 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.05119997262954712, "packed_extra_cold_setup_ms": -15.216941999824485, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.05221300307312049, "packed_extra_cold_setup_ms": -15.216941999824485, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.029695987701416016, "packed_extra_cold_setup_ms": -15.216941999824485, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.030793999030720443, "packed_extra_cold_setup_ms": -15.216941999824485, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
