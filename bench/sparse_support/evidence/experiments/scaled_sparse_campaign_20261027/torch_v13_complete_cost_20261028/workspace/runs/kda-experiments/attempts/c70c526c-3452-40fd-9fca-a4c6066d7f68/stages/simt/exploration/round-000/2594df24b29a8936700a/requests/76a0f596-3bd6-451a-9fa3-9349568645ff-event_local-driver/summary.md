# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.947883 ms, packed 1.028096 ms (paired SIMT/packed 0.921, 95% bootstrap [0.9191078618164107, 0.9249499390819359]); synchronized wall SIMT 0.953632 ms, packed 1.033176 ms (ratio 0.921).
- training: CUDA event SIMT 6.728704 ms, packed 6.920192 ms (paired SIMT/packed 0.972, 95% bootstrap [0.971541560685008, 0.9745883892203614]); synchronized wall SIMT 6.741706 ms, packed 6.933658 ms (ratio 0.973).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.750016 ms, packed-owner median 1.757867 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.823 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.519 ms; CE-GEO host compilation 0.078 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.889 ms (p10-p90 39.596-57.576); packed median 42.046 ms (p10-p90 40.052-42.337); packed extra including CE-GEO host compile 1.235 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08021330833435059, "packed_extra_cold_setup_ms": 1.2353839993011206, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07954466733887477, "packed_extra_cold_setup_ms": 1.2353839993011206, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19148826599121094, "packed_extra_cold_setup_ms": 1.2353839993011206, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.19195199881020564, "packed_extra_cold_setup_ms": 1.2353839993011206, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
