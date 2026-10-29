# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### event_local
- Shape: `{"cells": 4, "event_occurrences_per_cell": 1, "relation_width": 64, "selected_entities_per_cell": 1}`
- Work: 8 primal / 16 derivative candidates; 50.0% derivative-only.
- inference: CUDA event SIMT 0.960512 ms, packed 1.037056 ms (paired SIMT/packed 0.926, 95% bootstrap [0.9218981532260394, 0.9304733625910054]); synchronized wall SIMT 0.967541 ms, packed 1.045314 ms (ratio 0.926).
- training: CUDA event SIMT 4.100864 ms, packed 4.108800 ms (paired SIMT/packed 0.998, 95% bootstrap [0.9790595999177555, 1.0159690385384568]); synchronized wall SIMT 4.109000 ms, packed 4.116764 ms (ratio 0.998).
- Cold setup: SIMT 516.491 ms; packed 498.340 ms including shared geometry compilation; packed extra -18.152 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -18.151547003071755, "training_step_saving_ms": -0.00793600082397461}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -18.151547003071755, "training_step_saving_ms": -0.007764499059703667}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 8, 'packed': 8}.
