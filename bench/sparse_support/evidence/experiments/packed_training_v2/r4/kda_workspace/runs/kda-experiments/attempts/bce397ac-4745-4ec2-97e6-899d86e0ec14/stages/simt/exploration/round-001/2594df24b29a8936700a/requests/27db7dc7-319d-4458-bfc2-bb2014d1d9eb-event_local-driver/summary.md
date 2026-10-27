# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.845568 ms, packed 0.915200 ms (paired SIMT/packed 0.924, 95% bootstrap [0.9010561633081159, 0.94707206291881]); synchronized wall SIMT 0.853414 ms, packed 0.923005 ms (ratio 0.925).
- training: CUDA event SIMT 3.513344 ms, packed 3.717888 ms (paired SIMT/packed 0.943, 95% bootstrap [0.8734213391551898, 1.0124398084834068]); synchronized wall SIMT 3.521268 ms, packed 3.725803 ms (ratio 0.943).
- Cold setup: SIMT 522.236 ms; packed 474.078 ms including shared geometry compilation; packed extra -48.158 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -48.15789200074505, "training_step_saving_ms": -0.20454394817352295}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -48.15789200074505, "training_step_saving_ms": -0.2045344999714871}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
