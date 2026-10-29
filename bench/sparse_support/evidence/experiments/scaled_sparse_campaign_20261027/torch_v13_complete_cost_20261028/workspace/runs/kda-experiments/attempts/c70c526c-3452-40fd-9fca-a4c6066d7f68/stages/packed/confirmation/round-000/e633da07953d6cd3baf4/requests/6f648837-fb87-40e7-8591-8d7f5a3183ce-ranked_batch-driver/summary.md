# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.936277 ms, packed 1.018197 ms (paired SIMT/packed 0.918, 95% bootstrap [0.9081461614514709, 0.9242423969232488]); synchronized wall SIMT 0.941504 ms, packed 1.023476 ms (ratio 0.918).
- training: CUDA event SIMT 7.294635 ms, packed 7.449942 ms (paired SIMT/packed 0.987, 95% bootstrap [0.9578983470325596, 0.9977913604396915]); synchronized wall SIMT 7.307074 ms, packed 7.463196 ms (ratio 0.987).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.947648 ms, packed-owner median 1.943893 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 226.888 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 96.793 ms; CE-GEO host compilation 0.112 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 40.615 ms (p10-p90 40.202-40.888); packed median 40.557 ms (p10-p90 40.250-40.894); packed extra including CE-GEO host compile 0.054 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08191998799641931, "packed_extra_cold_setup_ms": 0.05388799763750285, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08197266652132373, "packed_extra_cold_setup_ms": 0.05388799763750285, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.15530713399251272, "packed_extra_cold_setup_ms": 0.05388799763750285, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.15612199907385094, "packed_extra_cold_setup_ms": 0.05388799763750285, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
