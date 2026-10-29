# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 1.015808 ms, packed 1.075200 ms (paired SIMT/packed 0.945, 95% bootstrap [0.944761925669147, 0.944761925669147]); synchronized wall SIMT 1.032347 ms, packed 1.091562 ms (ratio 0.946).
- training: CUDA event SIMT 3.336192 ms, packed 3.411968 ms (paired SIMT/packed 0.978, 95% bootstrap [0.977791087044491, 0.977791087044491]); synchronized wall SIMT 3.352005 ms, packed 3.427597 ms (ratio 0.978).
- Preparation: common CUDA/Torch initialization 210.253 ms; fixture owner setup 513.955 ms; CE-GEO host compilation 0.141 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 494.091 ms (p10-p90 494.091-494.091); packed median 472.379 ms (p10-p90 472.379-472.379); packed extra including CE-GEO host compile -21.571 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.05939197540283203, "packed_extra_cold_setup_ms": -21.57111199994688, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.05921499905525707, "packed_extra_cold_setup_ms": -21.57111199994688, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0757761001586914, "packed_extra_cold_setup_ms": -21.57111199994688, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07559199730167165, "packed_extra_cold_setup_ms": -21.57111199994688, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
