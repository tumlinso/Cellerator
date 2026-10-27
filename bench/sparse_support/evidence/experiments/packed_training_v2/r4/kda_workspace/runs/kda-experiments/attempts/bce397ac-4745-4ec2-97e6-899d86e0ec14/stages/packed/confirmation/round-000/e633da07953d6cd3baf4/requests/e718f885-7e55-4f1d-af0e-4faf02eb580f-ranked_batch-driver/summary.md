# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.867584 ms, packed 0.928768 ms (paired SIMT/packed 0.934, 95% bootstrap [0.9305785597865809, 0.937672403679803]); synchronized wall SIMT 0.875641 ms, packed 0.936770 ms (ratio 0.935).
- training: CUDA event SIMT 3.864832 ms, packed 4.216576 ms (paired SIMT/packed 0.914, 95% bootstrap [0.7983359938419143, 1.0296982141021793]); synchronized wall SIMT 3.872618 ms, packed 4.224688 ms (ratio 0.914).
- Cold setup: SIMT 524.261 ms; packed 472.051 ms including shared geometry compilation; packed extra -52.211 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -52.210717996786116, "training_step_saving_ms": -0.3517441749572754}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -52.210717996786116, "training_step_saving_ms": -0.3520702512105345}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
