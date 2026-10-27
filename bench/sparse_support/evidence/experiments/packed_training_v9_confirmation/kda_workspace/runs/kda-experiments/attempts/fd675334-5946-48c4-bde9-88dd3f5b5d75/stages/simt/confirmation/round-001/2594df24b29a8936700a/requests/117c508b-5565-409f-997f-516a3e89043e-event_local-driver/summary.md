# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.928085 ms, packed 1.004885 ms (paired SIMT/packed 0.922, 95% bootstrap [0.9198793276899805, 0.9242526959004091]); synchronized wall SIMT 0.933916 ms, packed 1.010580 ms (ratio 0.923).
- training: CUDA event SIMT 7.299754 ms, packed 7.483733 ms (paired SIMT/packed 0.975, 95% bootstrap [0.968393717258239, 0.9780365241419686]); synchronized wall SIMT 7.312114 ms, packed 7.497386 ms (ratio 0.975).
- Preparation: common CUDA/Torch initialization 215.978 ms; fixture owner setup 512.406 ms; CE-GEO host compilation 0.142 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 467.337 ms (p10-p90 466.907-494.147); packed median 468.552 ms (p10-p90 465.889-469.637); packed extra including CE-GEO host compile 1.357 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07680002848307299, "packed_extra_cold_setup_ms": 1.3569319999078289, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07666400051675748, "packed_extra_cold_setup_ms": 1.3569319999078289, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18397903442382812, "packed_extra_cold_setup_ms": 1.3569319999078289, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18527266729506664, "packed_extra_cold_setup_ms": 1.3569319999078289, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
