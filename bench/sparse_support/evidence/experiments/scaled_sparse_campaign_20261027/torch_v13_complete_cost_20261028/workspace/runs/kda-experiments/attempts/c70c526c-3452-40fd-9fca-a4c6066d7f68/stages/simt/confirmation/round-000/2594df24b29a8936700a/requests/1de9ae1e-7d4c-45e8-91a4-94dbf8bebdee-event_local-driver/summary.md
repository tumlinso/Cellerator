# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.941397 ms, packed 1.015808 ms (paired SIMT/packed 0.923, 95% bootstrap [0.9194359886915746, 0.925318204487378]); synchronized wall SIMT 0.946796 ms, packed 1.022374 ms (ratio 0.924).
- training: CUDA event SIMT 7.146837 ms, packed 7.302485 ms (paired SIMT/packed 0.982, 95% bootstrap [0.9645078331264987, 0.9934561027270504]); synchronized wall SIMT 7.160046 ms, packed 7.315884 ms (ratio 0.982).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.010453 ms, packed-owner median 1.912149 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 209.256 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.203 ms; CE-GEO host compilation 0.077 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.539 ms (p10-p90 40.030-41.005); packed median 40.531 ms (p10-p90 40.334-42.142); packed extra including CE-GEO host compile 0.069 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07441067695617687, "packed_extra_cold_setup_ms": 0.06919300358276814, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07557766730315041, "packed_extra_cold_setup_ms": 0.06919300358276814, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.15564791361490915, "packed_extra_cold_setup_ms": 0.06919300358276814, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.15583866722105721, "packed_extra_cold_setup_ms": 0.06919300358276814, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
