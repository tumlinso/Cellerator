# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.967680 ms, packed 1.026048 ms (paired SIMT/packed 0.943, 95% bootstrap [0.9431138009790506, 0.9431138009790506]); synchronized wall SIMT 0.982389 ms, packed 1.041815 ms (ratio 0.943).
- training: CUDA event SIMT 3.436544 ms, packed 3.577856 ms (paired SIMT/packed 0.961, 95% bootstrap [0.9605036871737106, 0.9605036871737106]); synchronized wall SIMT 3.451816 ms, packed 3.593453 ms (ratio 0.961).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.013792 ms, packed-owner median 0.945152 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 216.837 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 83.611 ms; CE-GEO host compilation 0.078 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 40.685 ms (p10-p90 40.685-40.685); packed median 44.920 ms (p10-p90 44.920-44.920); packed extra including CE-GEO host compile 4.313 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.05836796760559082, "packed_extra_cold_setup_ms": 4.312914996262407, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.059426001826068386, "packed_extra_cold_setup_ms": 4.312914996262407, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.1413121223449707, "packed_extra_cold_setup_ms": 4.312914996262407, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.14163700325298123, "packed_extra_cold_setup_ms": 4.312914996262407, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
