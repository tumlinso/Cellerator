# Native matrix and process operations

`Cellerator::matrix_patch` exposes `tanh(L X) R` through
`<Cellerator/math/matrix/patch.hh>`. It compiles the completed native patch
provider unchanged and adds public span/axis/policy/generation/alias admission.
Square contiguous row-major FP32 state and factors are supported. Four named
input/output axes preserve both mathematical sides of the patch; actor-private
columns acquire no shared biological meaning. Strides, rectangular shapes,
FP16/FP64, CUDA and capture are explicitly unsupported in this host adapter.

`Cellerator::private_ports` exposes `D_i sum_edges(dst=i) w_edge E_src h_src`
through `<Cellerator/math/process/ports.hh>`. It compiles the completed native
transport provider unchanged. Actors retain explicit private axes and ragged
widths; the port axis and ordered directed edge axis are separate. Each edge
retains its own coefficient and gradient, including repeated pairs and zeros.
Both components execute forward, input/parameter VJP and combined JVP. Named
parameter roles may borrow the same master storage; derivative outputs remain
separate contributions for the caller's existing owner to assemble. No implicit
parameter tying, averaging, owner publication or optimizer is introduced.

Borrowed tapes retain primal, topology, workspace and live generation metadata.
The external owner serializes changes, keeps bytes unchanged through responses,
and advances the corresponding native generations on publication. The wrappers
reject stale generations, bad extents/axes/policies and output aliases before
writes. They do not supply CUDA reader leases or immutable snapshots. Patch
forward uses caller scratch; its derivatives allocate in the existing provider.
Port calls allocate temporary port arrays in the existing provider. Capability
records report these limits; no allocation-free/capture claim is made.

## M01–M06 disposition

| Card | Callable owner or preserved alternative | Current boundary |
| --- | --- | --- |
| M01 patch | `matrix/patch.hh`; original `experiments/moonshot-parallel-v1/diff/patch/native.cc` | FP32 contiguous square nonlinear patch, forward/VJP/JVP; no arbitrary sparse lowering or decay interpretation. Plain LXR is a distinct restricted family. |
| M02 polynomial | `experiments/moonshot-parallel-v1/trajectory/polynomial.py`; `trajectory/test_polynomial.py` | Existing Torch polynomial forward/state+factor derivatives retained; native promotion belongs to MATH/DIFF. No new native polynomial claim. |
| M03 Sylvester | `experiments/moonshot-parallel-v1/trajectory/flow.py`; `trajectory/test_flow.py` | Existing differentiable Torch exponential route retained; CE MATH owns future generator/dt/cache integration. No detached generator or stale-cache shortcut. |
| M04 micro-MMA | `experiments/moonshot-parallel-v1/mma/quad/quad_mma.cuh`, `quad_mma.cu`; `quad_test.cu` | Existing checked SM70 FP16-input/FP32-output provider retained; no CUDA execution or derivative grant from this host gate. |
| M05 private ports | `process/ports.hh`; original `experiments/moonshot-parallel-v1/diff/ports/transport.cc` | Ragged contiguous FP32 transport, forward/VJP/JVP; local nonlinear law and scientific port interpretation stay with caller. |
| M06 process product | `process/product.hh` aliases existing `compute/operation/product2/product2.hh` | Existing typed native prepared owner, FP32 forward/VJP/JVP, ordered/repeated arguments, canonical coefficients and generation admission. Link `Cellerator::product2`; no duplicate owner or new optimizer. Original SM70 process-packet candidate remains at `mma/product/product.cu` with its narrower receipt. |

Existing SM70 patch provider `mma/patch16/patch16.cuh` and product2 native CUDA
provider `src/compute/operation/product2/product2.cu` remain available through
their original owners and exact receipts. Reusing these providers requires their
actual prepared/device/precision/stream contracts. This change does not run GPU
code, install new candidates or waive native GPU acceptance.

## Accepted GH numerical requests

GH-MODELS-PATCH can select this host `tanh(L X) R` path only when its square,
contiguous FP32 constraints match. Its scientific `softplus(decay) * X` term,
layout and ordinary Torch autograd composition remain GH/framework concerns;
this gate does not replace the current Torch client.
GH-MODELS-PORT can select the native transport and all role derivatives for
supported ragged contiguous FP32 inputs. The local law/coordinatewise tanh stays
an explicit additional operation. GH-MODELS-READOUT keeps its current framework
reduction/VJPs; STATE provides separately declared native observable values and
future DIFF can bind the derivative request. GH-MODELS-SYLVESTER remains assigned
to MATH. GH-MODELS-ML2 retains indexed MechanismModule/SharedSupportRelation
canonical owners and sanctioned checkpoint restoration.

## Focused gate and integration

```sh
python3 -B tests/substrate/operators/check.py --build-dir /tmp/ce-is1-ops-host
```

The native consumer executes scalar and noncommuting square patches plus ragged
port transport against independently stated algebra. It checks all input and
parameter finite differences for ports, shared patch parameter multiplicity,
JVP/VJP duality, repeated zero-weight edge gradients, stale generations,
unsupported policy/layout, aliases, bad topology and empty edges. Product2 is
included through its actual public owner; fresh product2 and GPU qualification
remain their original task's evidence, not inferred from this consumer.

Shared integration request: add `src/math/matrix` and `src/math/process` to the
selected root build, export the two component targets through BUILD, and route
any shared candidate registry changes through MERGE-A. Native implementation
sources remain in their preserved experimental paths; the supported public
entry path is the checked math header. No source outside this leaf was modified.
