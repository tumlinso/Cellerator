# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 1.004512 ms, packed 1.058816 ms (paired SIMT/packed 0.949, 95% bootstrap [0.9487125185431377, 0.9487125185431377]); synchronized wall SIMT 1.019675 ms, packed 1.074416 ms (ratio 0.949).
- training: CUDA event SIMT 3.722240 ms, packed 3.502080 ms (paired SIMT/packed 1.063, 95% bootstrap [1.0628654998628886, 1.0628654998628886]); synchronized wall SIMT 3.737832 ms, packed 3.516728 ms (ratio 1.063).
- Cold setup: SIMT 513.014 ms; packed 474.545 ms including shared geometry compilation; packed extra -38.469 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": -174.73033563316088, "extra_cold_setup_ms": -38.46863199942163, "training_step_saving_ms": 0.22016000747680664}, "synchronized_wall": {"break_even_steps": -173.98433240882738, "extra_cold_setup_ms": -38.46863199942163, "training_step_saving_ms": 0.22110400095698424}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
