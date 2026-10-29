# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.976213 ms, packed 1.054379 ms (paired SIMT/packed 0.924, 95% bootstrap [0.922582725372101, 0.9285014577569736]); synchronized wall SIMT 0.981829 ms, packed 1.059717 ms (ratio 0.925).
- training: CUDA event SIMT 6.864555 ms, packed 7.061163 ms (paired SIMT/packed 0.972, 95% bootstrap [0.9720228430517396, 0.9734132559066522]); synchronized wall SIMT 6.877568 ms, packed 7.074069 ms (ratio 0.973).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.744896 ms, packed-owner median 1.807019 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 209.372 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.764 ms; CE-GEO host compilation 0.091 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.549 ms (p10-p90 39.868-41.075); packed median 40.532 ms (p10-p90 40.214-41.842); packed extra including CE-GEO host compile 0.075 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0781653722127279, "packed_extra_cold_setup_ms": 0.07456500316038728, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07788866666184424, "packed_extra_cold_setup_ms": 0.07456500316038728, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.1966082255045576, "packed_extra_cold_setup_ms": 0.07456500316038728, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.19650100027016926, "packed_extra_cold_setup_ms": 0.07456500316038728, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
