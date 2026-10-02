# Transferable source

`cellerator/` contains native representation/kernel seeds and NumPy references. `glasshelix/` contains ordinary Torch model examples. Copy through the helpers into new claimed experimental directories, then integrate using INTEGRATION.md.

CPU examples need Python3.10+, NumPy, and Torch for the GH examples. Reuse the existing project environment; this package does not prescribe an upgrade to the CUDA/Torch stack.

From the archive root:

```sh
python3 payload/cellerator/reference/smoke.py
python3 payload/glasshelix/smoke.py
python3 payload/glasshelix/learn_toy.py --family patch
python3 payload/glasshelix/learn_toy.py --family ports
cmake -S payload/cellerator -B /tmp/ce-moonshot-host
cmake --build /tmp/ce-moonshot-host
/tmp/ce-moonshot-host/moonshot_layout
```

Optional, unrun in this delivery, using the existing CUDA12 toolchain and an assigned V100:

```sh
cmake -S payload/cellerator -B /tmp/ce-moonshot-cuda -DMOONSHOT_CUDA=ON
cmake --build /tmp/ce-moonshot-cuda
/tmp/ce-moonshot-cuda/moonshot_gpu_smoke
```

Use fresh build directories. The GPU smoke currently exercises only the quad-MMA mapping. Add patch and process same-policy checks while integrating. No command above is an instruction to run an unassigned benchmark or modify a live repository outside a claim.
