# 4. Validation, execution evidence and the small program

## Evidence levels must stay separate

A schema that parses, an interface that compiles, a test that computes reference mathematics, a GPU kernel that executes, and a speedup measured on a workload are five different results. None substitutes for the next. Current package checks validate planning consistency; they do not validate future Cellerator implementation.

The required integrated targets are proposed names to be wired by I04:

| Target | What it must actually test |
|---|---|
| `ceSpineCoreTest` | Canonical validation/equivalence, direction, identities and metadata rejection |
| `ceSpineAlgebraTest` | Dot split-K sums versus edge-channel concatenation; no false primitive lowering |
| `ceSpineFrontendTest` | Real parser/Sema/IR-origin descriptors, altered-source and diagnostics tests |
| `ceSpineNativeTest` | Existing device implementations, preparation, refresh, both orientations, failure preservation |
| `ceSpineCrossOriginTest` | Native and source-derived descriptors submitted through the same GPU path |
| `ceSemanticSpineDemo` | The exact user-supplied small biological program, in normal CUDA mode |

GPU targets accept `--require-sm70` and fail when a suitable actual device is absent. Do not translate missing CUDA into a CTest skip that then passes the epic. The default example's terminal output is useful for a human, but a success phrase alone is not a test.

## Required coverage

Use a non-square, asymmetric support relation and another structurally different one. Exercise empty rows/support, changed values, independent edge identities, wrong biological axes at equal extents, high-bit-only identity changes, stale structure/value/order, pointer rebinding, unsupported width/precision, aliasing and checked index conversion. Duplicate endpoint contributions must either execute exactly as represented or be explicitly rejected by a limited provider, never silently merged with changed rounding or identity.

Forward and transpose have independent logical-edge oracles. Check the adjoint identity within justified tolerance as a useful additional relation, not as a replacement for both reference outputs. Test the contraction law at K=2 and K=17 to distinguish a contracted dimension from an output channel. Change source spelling/binding in the compiler test so it cannot pass by returning one canned descriptor.

Required failing controls include wrong-index output, stale generation, NaN compared with finite expected output, missing implementation and missing device. Empty-math cases still obey the output/update contract and do not access null storage when no element is required.

## Demo content and exact placement

Place all four files in **`examples/semantic_spine_v1/`**. The program is `regulatory_reuse.cc`; `fixture.hh` defines its tiny inputs. `CMakeLists.txt` is repository-local integration which I04 enables, not an installed SDK consumer.

The fixture has four regulator identities, five gene identities and nine signed edges. Gene-D has no incoming edge. It is deliberately illustrative, not a fitted or biologically validated model. It runs:

1. One shared topology preparation for forward and transpose.
2. Generation 1 with two forward input states and one transpose gene signal.
3. Generation 2 with changed weights and another forward/transpose pair.
4. A deliberately stale generation-1 call, which must fail before touching sentinel output.

The normal program checks three forward launches, two transpose launches, two value refreshes and one topology preparation. It prints the real bound candidate names. It does not contain a hidden implementation of the GPU operation and cannot pass by falling back to its oracle.

Expected first-generation results for state A are `[0, 1.25, 0.5, 0, 1.5]`; the transpose signal gives `[0.75, -4, 0.75, 0.125]`. At generation 2 the corresponding outputs are `[1, 0.75, -0.5, 0, 1.5]` and `[1.25, -3, 0.75, -0.125]`. These are derived from the provided toy inputs, not external biological data.

## What can run before implementation

A standalone reference-only build checks the fixture and hard-coded half encodings without Cellerator or CUDA:

```bash
c++ -std=c++17 -Wall -Wextra -pedantic   -DCELLERATOR_SPINE_REFERENCE_ONLY=1   examples/semantic_spine_v1/regulatory_reuse.cc -o /tmp/ce-spine-reference
/tmp/ce-spine-reference
```

It prints `REFERENCE_ONLY_FIXTURE_PASS` and explicitly says no Cellerator/GPU execution was tested. It is excluded from the required hardware gate. This mode is provided for early sanity checking, not to weaken acceptance.

The normal demo cannot build against the reviewed baseline because the proposed native pair API is not yet implemented. That is expected: it is an executable acceptance target for the epic, not a claim that those functions already exist.

## Post-implementation build and run

The existing root build is CUDA-centric below its host-only early return [S03]. The small semantic tests can have standalone host builds; do not reorganize the full host SDK just to run them. The integrated suite uses the real CUDA build with a suitable CUDA 12.x compiler. CUDA 13 removed offline Volta compilation/library support [R03]. Supply the actually installed toolchain via the project's existing environment/settings rather than inventing a compiler path.

```bash
cmake -S . -B build-ss1   -DCELLERATOR_BUILD_TESTS=ON   -DCELLERATOR_BUILD_SEMANTIC_SPINE_V1=ON   -DCELLERATOR_ENABLE_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES=70   -DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF
cmake --build build-ss1 --parallel 4 --target ceSemanticSpineDemo
./build-ss1/ceSemanticSpineDemo --require-sm70
```

`CELLERATOR_BUILD_SEMANTIC_SPINE_V1` is the proposed opt-in switch I02/I04 add, not an existing baseline feature. The supplied `scripts/run_gate.py` runs actual build/execution commands and Compute Sanitizer for the demo/final groups. It requires the execution owner to acquire the canonical GPU lease first.

## Performance observations, not a new optimization campaign

Record cold topology preparation, value refresh and resident repeated execution separately, plus copies and synchronization. Compare against the same existing direct candidate with matching numeric tuple and timing boundaries to expose accidental host fallback or repeated topology work. Do not impose an arbitrary speedup threshold, claim broad superiority, or turn a tiny fixture into a research benchmark. A material regression caused by the adapter should be diagnosed and corrected within scope; an existing slow kernel is a later algorithmic task.

No hidden hot-path allocation, geometry search, host math, full support hashing or global synchronization may be disguised as “semantic validation.” Cold preparation and caller readback are explicit. Record exact hardware, compiler/build flags, source revision, candidate identity, tolerance and test commands. Evidence must originate from the actual run, not from a pre-generated acceptance JSON.
