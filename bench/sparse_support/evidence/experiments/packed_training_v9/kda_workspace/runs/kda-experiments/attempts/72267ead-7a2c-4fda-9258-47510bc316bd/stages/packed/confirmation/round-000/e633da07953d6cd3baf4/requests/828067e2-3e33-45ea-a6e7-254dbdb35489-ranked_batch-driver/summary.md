# Packed versus SIMT sparse-support training benchmark



Source identity: `72bcf0bea9f119ec00b739523fdad73af6c4f9a036536de40804a5da87f1dade`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.998400 ms, packed 1.060608 ms (paired SIMT/packed 0.941, 95% bootstrap [0.9261252503584442, 0.9561696018784409]); synchronized wall SIMT 1.006148 ms, packed 1.068465 ms (ratio 0.941).
- training: CUDA event SIMT 4.189440 ms, packed 4.437248 ms (paired SIMT/packed 0.942, 95% bootstrap [0.8887823515715845, 0.9955501864147499]); synchronized wall SIMT 4.197113 ms, packed 4.444979 ms (ratio 0.942).
- Cold setup: SIMT 515.657 ms; packed 475.547 ms including shared geometry compilation; packed extra -40.110 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -40.11044400249375, "training_step_saving_ms": -0.24780786037445068}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -40.11044400249375, "training_step_saving_ms": -0.24786599988146918}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
