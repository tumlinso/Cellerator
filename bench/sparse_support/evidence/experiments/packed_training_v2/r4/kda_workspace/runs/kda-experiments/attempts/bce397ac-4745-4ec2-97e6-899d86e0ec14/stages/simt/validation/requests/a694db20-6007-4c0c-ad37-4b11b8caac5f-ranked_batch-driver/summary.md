# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.903168 ms, packed 0.960512 ms (paired SIMT/packed 0.940, 95% bootstrap [0.9402985482152743, 0.9402985482152743]); synchronized wall SIMT 0.919593 ms, packed 0.976964 ms (ratio 0.941).
- training: CUDA event SIMT 2.937856 ms, packed 3.073024 ms (paired SIMT/packed 0.956, 95% bootstrap [0.9560146376845664, 0.9560146376845664]); synchronized wall SIMT 2.954267 ms, packed 3.089137 ms (ratio 0.956).
- Cold setup: SIMT 531.900 ms; packed 479.212 ms including shared geometry compilation; packed extra -52.689 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -52.688583000417566, "training_step_saving_ms": -0.13516807556152344}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -52.688583000417566, "training_step_saving_ms": -0.13487000251188874}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
