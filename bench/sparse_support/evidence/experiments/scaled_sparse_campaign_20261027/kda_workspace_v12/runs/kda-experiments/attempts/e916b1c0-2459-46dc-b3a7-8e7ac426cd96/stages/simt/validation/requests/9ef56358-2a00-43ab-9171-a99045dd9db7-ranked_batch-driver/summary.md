# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.002496 ms, packed 1.079296 ms (paired SIMT/packed 0.929, 95% bootstrap [0.9288425146990913, 0.9288425146990913]); synchronized wall SIMT 1.018362 ms, packed 1.097291 ms (ratio 0.928).
- training: CUDA event SIMT 3.547136 ms, packed 4.077568 ms (paired SIMT/packed 0.870, 95% bootstrap [0.8699146209690245, 0.8699146209690245]); synchronized wall SIMT 3.564386 ms, packed 4.096225 ms (ratio 0.870).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 0.926720 ms, packed-owner median 1.296384 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 1560.718 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 524.396 ms; CE-GEO host compilation 0.092 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 470.544 ms (p10-p90 470.544-470.544); packed median 467.364 ms (p10-p90 467.364-467.364); packed extra including CE-GEO host compile -3.087 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07679998874664307, "packed_extra_cold_setup_ms": -3.0870219961798284, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07892899884609506, "packed_extra_cold_setup_ms": -3.0870219961798284, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.5304319858551025, "packed_extra_cold_setup_ms": -3.0870219961798284, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.5318390030879527, "packed_extra_cold_setup_ms": -3.0870219961798284, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
