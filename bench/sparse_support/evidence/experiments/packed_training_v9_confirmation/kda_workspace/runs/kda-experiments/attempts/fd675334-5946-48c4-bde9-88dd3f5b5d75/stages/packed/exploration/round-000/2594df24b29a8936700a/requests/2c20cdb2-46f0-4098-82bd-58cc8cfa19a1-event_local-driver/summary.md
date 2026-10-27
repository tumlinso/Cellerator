# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.971776 ms, packed 1.036629 ms (paired SIMT/packed 0.921, 95% bootstrap [0.9059716028042917, 0.9273687539279122]); synchronized wall SIMT 0.976924 ms, packed 1.042236 ms (ratio 0.922).
- training: CUDA event SIMT 6.969002 ms, packed 7.132842 ms (paired SIMT/packed 0.977, 95% bootstrap [0.9725210570419645, 0.9859507857980008]); synchronized wall SIMT 6.981344 ms, packed 7.145281 ms (ratio 0.977).
- Preparation: common CUDA/Torch initialization 215.563 ms; fixture owner setup 513.081 ms; CE-GEO host compilation 0.153 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 470.604 ms (p10-p90 463.957-476.717); packed median 468.872 ms (p10-p90 461.011-483.077); packed extra including CE-GEO host compile -1.579 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.0648533503214519, "packed_extra_cold_setup_ms": -1.5788530035933945, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.0653123339967957, "packed_extra_cold_setup_ms": -1.5788530035933945, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.16383997599283795, "packed_extra_cold_setup_ms": -1.5788530035933945, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.1639366661644699, "packed_extra_cold_setup_ms": -1.5788530035933945, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
