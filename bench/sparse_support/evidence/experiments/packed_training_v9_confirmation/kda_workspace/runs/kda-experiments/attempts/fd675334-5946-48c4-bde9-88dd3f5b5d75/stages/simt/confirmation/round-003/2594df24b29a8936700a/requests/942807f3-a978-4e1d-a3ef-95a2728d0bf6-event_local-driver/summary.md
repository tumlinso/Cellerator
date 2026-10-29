# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.939691 ms, packed 1.028779 ms (paired SIMT/packed 0.920, 95% bootstrap [0.9138276984791821, 0.9224831809165136]); synchronized wall SIMT 0.945332 ms, packed 1.033839 ms (ratio 0.919).
- training: CUDA event SIMT 7.443456 ms, packed 7.629824 ms (paired SIMT/packed 0.976, 95% bootstrap [0.9733083218102742, 0.9782132255060653]); synchronized wall SIMT 7.456743 ms, packed 7.642407 ms (ratio 0.976).
- Preparation: common CUDA/Torch initialization 211.983 ms; fixture owner setup 509.000 ms; CE-GEO host compilation 0.143 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 468.080 ms (p10-p90 466.283-496.323); packed median 467.334 ms (p10-p90 464.638-469.826); packed extra including CE-GEO host compile -0.603 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.08908796310424794, "packed_extra_cold_setup_ms": -0.6031570010236464, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.08850666684641817, "packed_extra_cold_setup_ms": -0.6031570010236464, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.18636767069498728, "packed_extra_cold_setup_ms": -0.6031570010236464, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.18566433330609744, "packed_extra_cold_setup_ms": -0.6031570010236464, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
