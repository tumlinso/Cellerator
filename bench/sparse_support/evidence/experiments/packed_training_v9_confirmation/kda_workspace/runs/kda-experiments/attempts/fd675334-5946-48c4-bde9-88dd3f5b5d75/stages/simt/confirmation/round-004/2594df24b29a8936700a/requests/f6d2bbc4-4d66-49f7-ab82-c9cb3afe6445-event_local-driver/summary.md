# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.946517 ms, packed 1.025365 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9183126878985042, 0.9282407327205052]); synchronized wall SIMT 0.951939 ms, packed 1.030797 ms (ratio 0.921).
- training: CUDA event SIMT 7.511040 ms, packed 7.727104 ms (paired SIMT/packed 0.973, 95% bootstrap [0.9696595233049872, 0.9781563125923062]); synchronized wall SIMT 7.524631 ms, packed 7.739307 ms (ratio 0.973).
- Preparation: common CUDA/Torch initialization 233.478 ms; fixture owner setup 512.638 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.515 ms (p10-p90 465.295-483.839); packed median 467.484 ms (p10-p90 464.266-474.715); packed extra including CE-GEO host compile -1.890 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0788480440775553, "packed_extra_cold_setup_ms": -1.890239003841998, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07885833474574622, "packed_extra_cold_setup_ms": -1.890239003841998, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.216064453125, "packed_extra_cold_setup_ms": -1.890239003841998, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.21467666738317348, "packed_extra_cold_setup_ms": -1.890239003841998, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
