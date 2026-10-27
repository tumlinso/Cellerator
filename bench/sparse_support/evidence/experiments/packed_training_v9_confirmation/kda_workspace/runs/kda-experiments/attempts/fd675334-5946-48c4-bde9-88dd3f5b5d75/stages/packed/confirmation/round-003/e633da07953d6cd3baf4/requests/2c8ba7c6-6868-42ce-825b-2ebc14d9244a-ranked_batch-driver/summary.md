# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.950955 ms, packed 1.021269 ms (paired SIMT/packed 0.919, 95% bootstrap [0.9177257476122009, 0.9329534361331785]); synchronized wall SIMT 0.956669 ms, packed 1.026506 ms (ratio 0.920).
- training: CUDA event SIMT 7.403520 ms, packed 7.597739 ms (paired SIMT/packed 0.975, 95% bootstrap [0.9633688303945225, 0.9761036408881315]); synchronized wall SIMT 7.415269 ms, packed 7.610237 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 211.850 ms; fixture owner setup 518.334 ms; CE-GEO host compilation 0.147 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.882 ms (p10-p90 466.564-498.661); packed median 467.963 ms (p10-p90 465.399-470.061); packed extra including CE-GEO host compile -1.772 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07031464576721191, "packed_extra_cold_setup_ms": -1.7724479985190555, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.06983666632246854, "packed_extra_cold_setup_ms": -1.7724479985190555, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19421895345052143, "packed_extra_cold_setup_ms": -1.7724479985190555, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1949679996566065, "packed_extra_cold_setup_ms": -1.7724479985190555, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
