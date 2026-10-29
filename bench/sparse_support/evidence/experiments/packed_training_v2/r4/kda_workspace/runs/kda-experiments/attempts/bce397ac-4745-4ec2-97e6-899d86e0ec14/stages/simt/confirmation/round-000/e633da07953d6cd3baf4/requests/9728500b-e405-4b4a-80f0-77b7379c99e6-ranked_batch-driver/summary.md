# Packed versus SIMT sparse-support training benchmark



Source identity: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`

Frozen Torch extension: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.867584 ms, packed 0.962048 ms (paired SIMT/packed 0.906, 95% bootstrap [0.8337544442535889, 0.9775154183223185]); synchronized wall SIMT 0.875344 ms, packed 0.970417 ms (ratio 0.906).
- training: CUDA event SIMT 3.243264 ms, packed 3.640064 ms (paired SIMT/packed 0.893, 95% bootstrap [0.8032583303416486, 0.9820813857908959]); synchronized wall SIMT 3.250952 ms, packed 3.647666 ms (ratio 0.893).
- Cold setup: SIMT 514.723 ms; packed 461.157 ms including shared geometry compilation; packed extra -53.566 ms.
- Training break-even: `{"cuda_event": {"break_even_steps": null, "extra_cold_setup_ms": -53.56587699861848, "training_step_saving_ms": -0.39680004119873047}, "synchronized_wall": {"break_even_steps": null, "extra_cold_setup_ms": -53.56587699861848, "training_step_saving_ms": -0.39671349986747373}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
