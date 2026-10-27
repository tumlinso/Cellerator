# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.977152 ms, packed 1.043712 ms (paired SIMT/packed 0.936, 95% bootstrap [0.9253294286570092, 0.9472386034111295]); synchronized wall SIMT 0.985495 ms, packed 1.052201 ms (ratio 0.937).
- training: CUDA event SIMT 3.627008 ms, packed 3.884544 ms (paired SIMT/packed 0.932, 95% bootstrap [0.8468371133162579, 1.0170455245854313]); synchronized wall SIMT 3.635074 ms, packed 3.892956 ms (ratio 0.932).
- Cold setup: SIMT 517.196 ms; packed 476.612 ms including shared geometry compilation; packed extra -40.584 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -40.58424200411537, "training_step_saving_ms": -0.2575359344482422}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -40.58424200411537, "training_step_saving_ms": -0.2578820012786309}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
