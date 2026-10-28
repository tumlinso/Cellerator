# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.934571 ms, packed 1.013760 ms (paired SIMT/packed 0.924, 95% bootstrap [0.9197302022886036, 0.9254995530250718]); synchronized wall SIMT 0.940410 ms, packed 1.019352 ms (ratio 0.925).
- training: CUDA event SIMT 6.614016 ms, packed 6.800725 ms (paired SIMT/packed 0.974, 95% bootstrap [0.9725457097833559, 0.9737925139770859]); synchronized wall SIMT 6.628521 ms, packed 6.813594 ms (ratio 0.974).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.766059 ms, packed-owner median 1.722368 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 213.602 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 85.109 ms; CE-GEO host compilation 0.080 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 41.950 ms (p10-p90 40.341-43.107); packed median 42.921 ms (p10-p90 40.353-51.664); packed extra including CE-GEO host compile 1.051 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07918938000996911, "packed_extra_cold_setup_ms": 1.0513970082683954, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07894133280691074, "packed_extra_cold_setup_ms": 1.0513970082683954, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18670908610025982, "packed_extra_cold_setup_ms": 1.0513970082683954, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1850729992535589, "packed_extra_cold_setup_ms": 1.0513970082683954, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
