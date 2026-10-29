# Packed versus SIMT sparse-support training benchmark



Source identity: `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`

Frozen Torch extension: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`



SIMT control and packed candidate use identical support, source inputs, workset rows, hard admission bits, and initial shared parameters. Packed backward is the existing SIMT VJP.



### ranked_batch
- Shape: `{"cells": 32, "relation_width": 64, "selected_genes_per_cell": 8}`
- Work: 449 primal / 896 derivative candidates; 49.9% derivative-only.
- inference: CUDA event SIMT 0.934912 ms, packed 1.014443 ms (paired SIMT/packed 0.922, 95% bootstrap [0.9175051016615771, 0.9240890402829222]); synchronized wall SIMT 0.940456 ms, packed 1.019681 ms (ratio 0.922).
- training: CUDA event SIMT 6.214656 ms, packed 6.143659 ms (paired SIMT/packed 0.979, 95% bootstrap [0.9651034705833086, 1.0021658077898281]); synchronized wall SIMT 6.224152 ms, packed 6.158704 ms (ratio 0.979).
- Preparation: common CUDA/Torch initialization 212.377 ms; fixture owner setup 535.566 ms; CE-GEO host compilation 0.143 ms.
- Balanced module setup (5 AB/BA pairs): SIMT median 469.420 ms (p10-p90 465.786-492.418); packed median 467.474 ms (p10-p90 465.150-489.912); packed extra including CE-GEO host compile -1.804 ms.
- Setup amortization estimates: `{"inference_cuda_event": {"break_even_steps": null, "median_step_saving_ms": -0.07953071594238281, "packed_extra_cold_setup_ms": -1.803547998861177, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "inference_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": -0.07922466708502418, "packed_extra_cold_setup_ms": -1.803547998861177, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_cuda_event": {"break_even_steps": null, "median_step_saving_ms": 0.07099723815917969, "packed_extra_cold_setup_ms": -1.803547998861177, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}, "training_synchronized_wall": {"break_even_steps": null, "median_step_saving_ms": 0.0654480002898099, "packed_extra_cold_setup_ms": -1.803547998861177, "positive_extra_setup_cost": false, "positive_step_saving_supported_by_paired_ci": false}}`
- Checks: output=True, gradients=True, derivative-only gate gradient counts={'simt': 447, 'packed': 447}.
