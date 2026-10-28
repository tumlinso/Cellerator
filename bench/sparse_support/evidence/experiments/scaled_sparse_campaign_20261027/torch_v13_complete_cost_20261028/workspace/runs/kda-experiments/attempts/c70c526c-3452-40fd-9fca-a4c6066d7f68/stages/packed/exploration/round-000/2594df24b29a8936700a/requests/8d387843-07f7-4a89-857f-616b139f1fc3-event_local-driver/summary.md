# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.932864 ms, packed 1.012053 ms (paired SIMT/packed 0.921, 95% bootstrap [0.9187183901971469, 0.9228948668440179]); synchronized wall SIMT 0.938368 ms, packed 1.017552 ms (ratio 0.922).
- training: CUDA event SIMT 6.706518 ms, packed 6.841003 ms (paired SIMT/packed 0.972, 95% bootstrap [0.9697516043953501, 0.9770437467183614]); synchronized wall SIMT 6.720026 ms, packed 6.854649 ms (ratio 0.972).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.784149 ms, packed-owner median 1.753088 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 227.084 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 87.603 ms; CE-GEO host compilation 0.079 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.647 ms (p10-p90 40.450-41.473); packed median 40.638 ms (p10-p90 40.340-41.099); packed extra including CE-GEO host compile 0.070 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07918930053710926, "packed_extra_cold_setup_ms": 0.06982599734328687, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07918433281398995, "packed_extra_cold_setup_ms": 0.06982599734328687, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.13448524475097656, "packed_extra_cold_setup_ms": 0.06982599734328687, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1346233342095129, "packed_extra_cold_setup_ms": 0.06982599734328687, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
