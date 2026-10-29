# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.932523 ms, packed 1.018880 ms (paired SIMT/packed 0.918, 95% bootstrap [0.9048885684338179, 0.9211058937474296]); synchronized wall SIMT 0.938000 ms, packed 1.024400 ms (ratio 0.919).
- training: CUDA event SIMT 7.406251 ms, packed 7.603541 ms (paired SIMT/packed 0.977, 95% bootstrap [0.972885618398881, 0.9788512654666511]); synchronized wall SIMT 7.418458 ms, packed 7.616410 ms (ratio 0.977).
- Preparation: common CUDA/Torch initialization 214.643 ms; fixture owner setup 511.538 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.573 ms (p10-p90 465.246-487.874); packed median 467.828 ms (p10-p90 467.681-477.993); packed extra including CE-GEO host compile -0.603 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08635727564493811, "packed_extra_cold_setup_ms": -0.6026170012773946, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08640033411211334, "packed_extra_cold_setup_ms": -0.6026170012773946, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.19729042053222656, "packed_extra_cold_setup_ms": -0.6026170012773946, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1979523331101518, "packed_extra_cold_setup_ms": -0.6026170012773946, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
