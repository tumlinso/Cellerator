# Experimental tensor numerical provider (CE-MOON-030)

Use target `Cellerator::moonshot_tensor`, header `<ce_moon/tensor.hpp>` and
namespace `ce_moon::tensor`. Each `Matrix16` has 256 row-major FP32 elements;
`TileShape{rows,features,outputs}` gives the active axes (1..16). Inactive
input elements are ignored and output padding is zero. Source identifiers
are opaque caller tokens. Baseplane owns their biological interpretation.

| Card | Host entry point | Semantics and fixture |
| --- | --- | --- |
| E21 | `object_features(x,w,shape,ids)` | Objects × features times features × output features. Returns values and IDs. Identity, nontrivial transform, and inactive padding are checked. |
| E22 | `relation_scores(q,k,rows,features)`, `compact_relations(scores,ids,rows,mask,threshold,capacity)` | Directed QKᵀ; explicit binary caller pair mask; stable row-major emit above threshold. Returns source IDs, row/column, scores, required count and overflow. Fixture uses IDs 7 and 9001, directed scores 7 and 6, and capacity 1. |
| E23 | `finite_relation_counts(a,b)`, `existence(counts)` | A single binary 16-state two-step product counts paths in [0,16]; positive counts mean existence. Multiple paths, empty rows and count16 are compared with integer Boolean composition. |
| E24 | `possible_states(states,w,shape)`, `interpolate(table,coordinates,rows,outputs,query)` | Shared region-conditioned transform followed by tanh, sampled entry-state rows and piecewise linear interpolation along an explicit scalar coordinate. For samples -1,+1 and query .5 the error against direct tanh is about .0813. No extrapolation. |

`<ce_moon/tensor.cuh>` provides corresponding asynchronous CUDA launch wrappers
in `ce_moon::tensor::cuda`. They use caller-owned nonaliasing device buffers,
caller stream, two FP16 packing tiles, a FP32 result tile, and explicit shape.
The WMMA buffers require 32-byte alignment and the target must support sm70.
CPU callers retain IDs for E21/E24; E22 device compaction accepts device IDs,
pair masks, capacity, pair output and required/overflow scalars. It uses a
bounded one-thread postpass to retain deterministic order on this tiny tile.
Wrappers perform no allocation or synchronization. Launch errors are returned;
completion errors belong to the caller's stream synchronization.

The CUDA caller must validate finite active values representable in FP16,
binary E23 inputs and binary E22 masks before launch. FP16 packing and FP32
accumulation differ from the host's FP32 inputs/double accumulation. E23 binary
counts are exactly representable; arbitrary weighted/repeated counting has no
such guarantee. Mask recovery, packing, provenance handling, sparse emission,
transfer and construction cost remain part of any future performance claim.
No throughput or biological efficacy has been measured.

Build from repository root with the repository's resource slot wrapper:

```sh
python /tmp/moonshot_build_slot.py cmake -S experiments/baseplane_moonshot -B /tmp/ce-moon-tensor-build-129 -DCE_MOON_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc
python /tmp/moonshot_build_slot.py cmake --build /tmp/ce-moon-tensor-build-129 --target ce_moon_tensor_host ce_moon_tensor_cuda -j1
ctest --test-dir /tmp/ce-moon-tensor-build-129 -R '^ce_moon_tensor_host$' --output-on-failure
```

The CUDA executable `families/tensor/ce_moon_tensor_cuda` is deliberately absent
from automatic CTest: only the controller may run it with assigned GPU resources.
It compares all four numerical paths with host oracles and checks directed
pair IDs/capacity on device. The provider receipt records current evidence;
compilation alone does not claim a GPU run. This family supplies E08/E10/E44/E46
with reusable numerical tiles but does not claim those scientific consumers.
