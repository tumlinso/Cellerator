# Grouped-linear operation record

- Name: `grouped_linear` / `GroupedLinear`.
- Purpose: trainable expert-major linear maps over compact ragged worklists.
- Owner: Cellerator NN foundation and reusable conditional modules.
- Status: implemented and numerically validated on CPU/reference and admitted
  V100 CUDA, including first-order autograd, composed training, and memcheck.
  See `REPORT.md` for exact evidence and limits.
- Python API: `cellerator.torch.grouped_linear.grouped_linear` and module.
- Binding: existing `cellerator._modeling_cuda.GroupedLinear` and `DeviceView`.
- Backend: Torch reference on CPU; existing cuBLAS FP16/FP32 CUDA operation.
- I/O: inputs `[N,I]`, shared weights `[E,O,I]`, host offsets `[E+1]`, output `[N,O]`.
- Layout/device: matching floating dtype/device; contiguous native operands;
  CUDA FP16/FP32 with FP32 accumulation; no implicit casts or transfers.
- Autograd: native first-order input/weight gradients; Torch owns canonical
  parameters, optimizer state, outputs, and gradient buffers.
- Distributed: local operator only; communication/routing are separate.
- Source: new `python/cellerator/torch/grouped_linear.py`; existing numerical
  implementation and binding remain unchanged.
