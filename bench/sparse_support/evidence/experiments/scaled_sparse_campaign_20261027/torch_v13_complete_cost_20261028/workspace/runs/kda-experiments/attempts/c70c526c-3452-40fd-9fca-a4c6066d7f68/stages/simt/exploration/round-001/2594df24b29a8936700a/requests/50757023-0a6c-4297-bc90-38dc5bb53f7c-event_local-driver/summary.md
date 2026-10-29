# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.968704 ms, packed 1.047552 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9171288525779442, 0.9292827579771424]); synchronized wall SIMT 0.974272 ms, packed 1.052935 ms (ratio 0.923).
- training: CUDA event SIMT 7.503531 ms, packed 7.703893 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9733067863113145, 0.9762879415143658]); synchronized wall SIMT 7.516962 ms, packed 7.716694 ms (ratio 0.975).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.185557 ms, packed-owner median 2.223787 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 213.392 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.949 ms; CE-GEO host compilation 0.079 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 41.767 ms (p10-p90 40.383-43.024); packed median 42.005 ms (p10-p90 40.481-42.571); packed extra including CE-GEO host compile 0.317 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07884804407755541, "packed_extra_cold_setup_ms": 0.3174330049660057, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0786629995369974, "packed_extra_cold_setup_ms": 0.3174330049660057, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.2003625233968096, "packed_extra_cold_setup_ms": 0.3174330049660057, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.19973199960077181, "packed_extra_cold_setup_ms": 0.3174330049660057, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
