# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.935936 ms, packed 1.016149 ms (paired SIMT/packed 0.920, 95% bootstrap [0.918120809878707, 0.9255813726827937]); synchronized wall SIMT 0.941392 ms, packed 1.021459 ms (ratio 0.921).
- training: CUDA event SIMT 7.459499 ms, packed 7.646549 ms (paired SIMT/packed 0.976, 95% bootstrap [0.9750380211598628, 0.9773185258329219]); synchronized wall SIMT 7.472830 ms, packed 7.659426 ms (ratio 0.976).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.254848 ms, packed-owner median 2.285227 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 211.605 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.446 ms; CE-GEO host compilation 0.078 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 42.339 ms (p10-p90 40.093-62.300); packed median 41.818 ms (p10-p90 40.539-43.784); packed extra including CE-GEO host compile -0.443 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08021338780721021, "packed_extra_cold_setup_ms": -0.44262299343245104, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08006666636598914, "packed_extra_cold_setup_ms": -0.44262299343245104, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18705050150553415, "packed_extra_cold_setup_ms": -0.44262299343245104, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18659633375743034, "packed_extra_cold_setup_ms": -0.44262299343245104, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
