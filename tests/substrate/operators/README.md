# Native matrix and process operations

`Cellerator::matrix_patch` exposes `tanh(L X) R` through
`<Cellerator/ops/matrix/patch.hh>`. It compiles the completed native patch
provider unchanged and adds public span/axis/policy/generation/alias admission.
Square contiguous row-major FP32 state and factors are supported. Four named
input/output axes preserve both mathematical sides of the patch; actor-private
columns acquire no shared biological meaning. Strides, rectangular shapes,
FP16/FP64, CUDA and capture are explicitly unsupported in this host adapter.

`Cellerator::private_ports` exposes `D_i sum_edges(dst=i) w_edge E_src h_src`
through `<Cellerator/ops/process/ports.hh>`. It compiles the completed native
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
| M01 patch | `matrix/patch.hh`; original `src/ops/matrix/patch_provider.cc` | FP32 contiguous square nonlinear patch, forward/VJP/JVP; no arbitrary sparse lowering or decay interpretation. Plain LXR is a distinct restricted family. |
| M02 polynomial | `Cellerator::frontier`; original `trajectory/polynomial.py` in the [Moonshot archive](../../../docs/history/baseplane.md) | Native host polynomial forward, JVP, and VJP have their own frontier checks. |
| M03 Sylvester | Original `trajectory/flow.py` and `trajectory/test_flow.py` in the [Moonshot archive](../../../docs/history/baseplane.md) | Differentiable Torch exponential reference for the declared local flow. |
| M04 micro-MMA | Original `mma/quad` candidate in the [Moonshot archive](../../../docs/history/baseplane.md) | Recorded SM70 FP16-input/FP32-output fixture checks. |
| M05 private ports | `process/ports.hh`; original `src/ops/process/ports_provider.cc` | Ragged contiguous FP32 transport, forward/VJP/JVP; local nonlinear law and scientific port interpretation stay with caller. |
| M06 process product | `compute/operation/product2/product2.hh` | Existing typed native prepared owner, FP32 forward/VJP/JVP, ordered/repeated arguments, canonical coefficients and generation admission. Link `Cellerator::product2`. The original SM70 process-packet candidate remains at `mma/product/product.cu` with its narrower receipt. |

The [Moonshot archive](../../../docs/history/baseplane.md) retains the SM70
patch candidate and its original receipts. The maintained native Product2 CUDA
provider is `src/compute/operation/product2/product2.cu`. This host operator gate
checks the selected host contracts.

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
unsupported policy/layout, aliases, bad topology and empty edges. The fixture
also checks that the canonical Product2 header compiles.

The standalone gate and normal root build use the native owners in
`src/ops/matrix` and `src/ops/process` directly.
