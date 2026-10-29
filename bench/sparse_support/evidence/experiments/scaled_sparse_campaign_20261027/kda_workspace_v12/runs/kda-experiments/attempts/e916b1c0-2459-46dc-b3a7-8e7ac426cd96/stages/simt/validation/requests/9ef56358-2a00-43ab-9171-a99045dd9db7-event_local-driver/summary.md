# Packed versus SIMT sparse-support training benchmark



Source identity: `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`

Frozen Torch extension: `6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.996352 ms, packed 1.046528 ms (paired SIMT/packed 0.952, 95% bootstrap [0.9520548272889822, 0.9520548272889822]); synchronized wall SIMT 1.011556 ms, packed 1.062261 ms (ratio 0.952).
- training: CUDA event SIMT 3.769344 ms, packed 3.878912 ms (paired SIMT/packed 0.972, 95% bootstrap [0.9717529344774735, 0.9717529344774735]); synchronized wall SIMT 3.785535 ms, packed 3.894772 ms (ratio 0.972).
- Global all-edge value refresh plus diagnostic all-ones VJP: SIMT-owner median 1.217536 ms, packed-owner median 1.205248 ms; the VJP cotangent is simpler than the relation-derived gradient, so this is phase attribution, not a substitute for complete training timing.
- Preparation: common CUDA/Torch initialization 228.425 ms; fixture bundle load 0.000 ms; support-spec construction 0.000 ms; fixture owner setup 520.141 ms; CE-GEO host compilation 0.080 ms.
- Balanced module setup (1 AB/BA pairs): SIMT median 450.852 ms (p10-p90 450.852-450.852); packed median 521.275 ms (p10-p90 521.275-521.275); packed extra including CE-GEO host compile 70.502 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.05017596483230591, "packed_extra_cold_setup_ms": 70.50208400323754, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.05070500264992006, "packed_extra_cold_setup_ms": 70.50208400323754, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.10956788063049316, "packed_extra_cold_setup_ms": 70.50208400323754, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.10923700028797612, "packed_extra_cold_setup_ms": 70.50208400323754, "positive_extra_setup_cost": true, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
