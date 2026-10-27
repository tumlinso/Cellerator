# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.878336 ms, packed 0.980224 ms (paired SIMT/packed 0.896, 95% bootstrap [0.8512740667341306, 0.9412382351656735]); synchronized wall SIMT 0.886268 ms, packed 0.988982 ms (ratio 0.896).
- training: CUDA event SIMT 3.828224 ms, packed 4.118016 ms (paired SIMT/packed 0.930, 95% bootstrap [0.849248737371547, 1.010207912288496]); synchronized wall SIMT 3.838123 ms, packed 4.123890 ms (ratio 0.931).
- Cold setup: SIMT 531.275 ms; packed 476.814 ms including shared geometry compilation; packed extra -54.461 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -54.46065399883082, "training_step_saving_ms": -0.28979194164276123}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -54.46065399883082, "training_step_saving_ms": -0.2857667495845817}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
