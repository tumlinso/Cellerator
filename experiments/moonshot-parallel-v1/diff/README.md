# Experimental native derivatives and Torch consumption

Run the complete CPU qualification from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/diff/check_diff.py
```

The NumPy interfaces in `patch/ops.py`, `product/ops.py` and `ports/ops.py`
compile and call the owned C++ CPU numerical implementations through ctypes.
The Torch consumer in `adapter/torch_ops.py` calls those same interfaces.
Native forward tapes own immutable copied operands and support explicit
value-generation checks. Torch saves original tensors to reject in-place
changes before backward. Backward returns gradients without an optimizer or
state update. Shared parameters and repeated indices accumulate their adjoints.

```python
# Add this diff directory to sys.path for the experimental import.
from adapter import patch, process, ports, patch_jvp
prediction = patch(x, left, right)  # tanh(left @ x) @ right
prediction.square().mean().backward()
# Ordinary torch.optim operates on left/right outside backward.
```

`patch` accepts matching nonempty square CPU float32 contiguous matrices.
`fp32` differentiates the FP32 expression. `stored_half_ste` rounds X, L,
R and the tanh intermediate to stored half values, performs the matrix
accumulations in FP32, and uses an explicit straight-through rounding
surrogate. Its derivatives are checked against that convention, rather
than a finite difference of the discontinuous rounding operation.

`process` accepts flattened float32 state/coefficient vectors and static
int64 ordered index vectors. Each output is `k[i]*x[a[i]]*x[b[i]]`.
Repeated slots and numerical zeros preserve derivative support.

`ports` accepts flattened heterogeneous actor states, encoders,
decoders and edge weights. Each actor's width is positive. A shared port
width P is inferred from encoder/state extents; encoders use actor-major
`[P,H_i]`, decoders `[H_i,P]` row-major storage. Static int64 src/dst vectors
specify supplied directed edges. The output is the transport contribution
`D_i sum_(dst=i) weight E_src h_src`, with no extra activation or residual.

All three operators expose input and parameter VJPs and explicit native
JVP helper functions. The Torch adapters support first-order reverse mode
only. CUDA derivatives, Torch forward AD, higher-order differentiation,
batching and production CelleraTorch registration are unsupported here.
The separate MMA lane qualifies CUDA forward primitives; those results do
not qualify these CPU derivative adapters for CUDA use.

`capability-receipt.json` binds the exact source set to the completed CPU
checks. The ordinary command verifies hashes and runs tests without changing
tracked evidence. After a reviewed source change, regenerate the receipt
explicitly with `--record`; that runs every check before recording it.
