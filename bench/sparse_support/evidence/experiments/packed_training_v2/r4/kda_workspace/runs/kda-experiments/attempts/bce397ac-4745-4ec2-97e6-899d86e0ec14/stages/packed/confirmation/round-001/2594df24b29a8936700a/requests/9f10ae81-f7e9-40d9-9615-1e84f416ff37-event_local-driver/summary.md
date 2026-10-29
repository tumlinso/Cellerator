# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.867840 ms, packed 0.947456 ms (paired SIMT/packed 0.916, 95% bootstrap [0.91431674273962, 0.9176090487740063]); synchronized wall SIMT 0.876093 ms, packed 0.955679 ms (ratio 0.917).
- training: CUDA event SIMT 3.733504 ms, packed 3.750144 ms (paired SIMT/packed 0.995, 95% bootstrap [0.916082985819491, 1.0741042277341484]); synchronized wall SIMT 3.741260 ms, packed 3.758313 ms (ratio 0.995).
- Cold setup: SIMT 540.813 ms; packed 506.150 ms including shared geometry compilation; packed extra -34.662 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -34.662088997720275, "training_step_saving_ms": -0.01663994789123535}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -34.662088997720275, "training_step_saving_ms": -0.01705249997030478}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
