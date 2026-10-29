# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.982699 ms, packed 1.058816 ms (paired SIMT/packed 0.926, 95% bootstrap [0.9221614712380756, 0.9336325917514351]); synchronized wall SIMT 0.988439 ms, packed 1.064509 ms (ratio 0.927).
- training: CUDA event SIMT 7.616512 ms, packed 7.811072 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9742689588512159, 0.9766811194092733]); synchronized wall SIMT 7.629790 ms, packed 7.824913 ms (ratio 0.975).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 2.277376 ms, packed-owner median 2.300587 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 208.161 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 80.270 ms; CE-GEO host compilation 0.086 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 41.597 ms (p10-p90 40.675-56.096); packed median 41.193 ms (p10-p90 40.765-44.584); packed extra including CE-GEO host compile -0.318 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07611735661824537, "packed_extra_cold_setup_ms": -0.3183920016454067, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07606999982575269, "packed_extra_cold_setup_ms": -0.3183920016454067, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19455973307291696, "packed_extra_cold_setup_ms": -0.3183920016454067, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.19512300180698094, "packed_extra_cold_setup_ms": -0.3183920016454067, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
