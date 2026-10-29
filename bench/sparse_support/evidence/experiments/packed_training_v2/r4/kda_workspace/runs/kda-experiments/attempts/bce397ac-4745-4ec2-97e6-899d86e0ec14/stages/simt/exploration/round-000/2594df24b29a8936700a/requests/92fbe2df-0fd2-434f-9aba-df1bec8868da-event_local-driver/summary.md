# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.890624 ms, packed 0.990720 ms (paired SIMT/packed 0.899, 95% bootstrap [0.8930912818248851, 0.9051403933579231]); synchronized wall SIMT 0.902790 ms, packed 0.999428 ms (ratio 0.903).
- training: CUDA event SIMT 3.852032 ms, packed 4.206592 ms (paired SIMT/packed 0.914, 95% bootstrap [0.8447932534562939, 0.9834602690728863]); synchronized wall SIMT 3.860633 ms, packed 4.217390 ms (ratio 0.914).
- Cold setup: SIMT 519.346 ms; packed 480.857 ms including shared geometry compilation; packed extra -38.489 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -38.489178998133866, "training_step_saving_ms": -0.35455989837646484}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -38.489178998133866, "training_step_saving_ms": -0.3567570011000498}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
