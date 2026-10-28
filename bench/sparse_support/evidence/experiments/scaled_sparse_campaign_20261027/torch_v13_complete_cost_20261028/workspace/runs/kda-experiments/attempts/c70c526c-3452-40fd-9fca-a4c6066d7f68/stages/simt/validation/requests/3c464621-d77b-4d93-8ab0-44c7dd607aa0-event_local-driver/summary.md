# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.989184 ms, packed 1.072128 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9226360741597806, 0.9226360741597806]); synchronized wall SIMT 1.004737 ms, packed 1.088228 ms (ratio 0.923).
- training: CUDA event SIMT 3.257344 ms, packed 3.464192 ms (paired SIMT/packed 0.940, 95% bootstrap [0.9402897093257656, 0.9402897093257656]); synchronized wall SIMT 3.273282 ms, packed 3.481149 ms (ratio 0.940).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 0.947200 ms, packed-owner median 0.957440 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 215.140 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 79.254 ms; CE-GEO host compilation 0.105 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 41.403 ms (p10-p90 41.403-41.403); packed median 40.901 ms (p10-p90 40.901-40.901); packed extra including CE-GEO host compile -0.397 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08294403553009033, "packed_extra_cold_setup_ms": -0.3968780038121622, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08349099880433641, "packed_extra_cold_setup_ms": -0.3968780038121622, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.2068479061126709, "packed_extra_cold_setup_ms": -0.3968780038121622, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.20786700042663142, "packed_extra_cold_setup_ms": -0.3968780038121622, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
