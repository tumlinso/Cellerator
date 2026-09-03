# Realization IR and backend plan

## Realization IR

Realization IR is deliberately physical. It represents:

- target and capability requirements;
- selected exact cover and contribution ownership;
- atoms, extents, local/global index spaces, and recovery maps;
- physical planes;
- projections and value-position maps;
- packed operands and invalidation;
- persistent order and order transforms;
- memory/workspace/residency requirements;
- prepared stage graphs;
- symbolic streams, readiness, publication, generations, and graph capture;
- typed runtime bindings;
- backend/native fragments;
- lowering-resumption artifacts.

Live pointers, CUDA events, streams, library handles, and CellShard leases remain launch/runtime bindings rather than persistent IR.

## Backend thin waist

Every backend provides target discovery, admissibility, code-generation plans, artifacts, diagnostics, source maps, toolchain identity, and optional native operations.

Backend specificity grows downward. Semantic IR remains toolchain-independent.

## CPU route

The first complete backend emits readable C++ and ordinary objects through GCC or Clang, including conventional fallbacks for core operation families.

## NVCC route

Cellerator emits CUDA or binds prelinked native providers. NVCC never parses Cellerator syntax. The driver owns host/device action graphs, device linking, fatbinaries, diagnostics, and ordinary object output.

## Clang CUDA and direct PTX

These are optional routes through the same backend ABI. Direct PTX is a legitimate typed Realization IR endpoint for extreme paths. It may remain experimental if complete-cost evidence does not justify promotion.

## Workstream task catalog

### F01: Realization IR

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-F01-001` | Freeze Realization IR module and target scopes | Define target-specific modules, functions, kernels, host stubs, data artifacts, stages, bindings, and native fragments while retaining source/semantic/planning lineage in cold metadata. |
| `CE-CCP1-F01-002` | Implement target and capability descriptions | Represent architecture class, compute capability, instruction families, collective scope, memory interfaces, numeric support, graph capture, toolchain, runtime, and backend requirements. |
| `CE-CCP1-F01-003` | Commit selected exact cover and contribution ownership | Materialize the selected atom/decomposition cover, owners, halos, replicas, partial contributors, canonical recovery, and certification receipt. |
| `CE-CCP1-F01-004` | Implement atom and extent bindings | Represent stable atom identity, physical instance role, global/local extents, local index width, multi-extent slices, alignment, address-space class, and recovery maps without requiring live addresses. |
| `CE-CCP1-F01-005` | Implement physical plane representation | Separate structure, values, active support, gradients, partials, workspace, metadata, and generated constants with independent lifetimes, generations, and residency requirements. |
| `CE-CCP1-F01-006` | Implement projection contracts | Represent CSR, feature-major, row-masked, dense fragment, MMA hybrid, transpose, vendor-specific, and extension projections with payload ABI and value-position maps. |
| `CE-CCP1-F01-007` | Implement packed operands and invalidation | Represent packed value/input/output operands, source generation, value-position maps, alignment, padding holes, pack operation, persistence horizon, and stale-generation behavior. |
| `CE-CCP1-F01-008` | Implement order transforms and persistent physical order | Represent logical, canonical, projection-native, and persistent physical orders plus explicit gather/scatter/canonicalize stages and reuse. |
| `CE-CCP1-F01-009` | Implement memory, workspace, and residency requirements | Represent persistent/transient/graph-stable allocations, alignment, capacity, lifetime, host/device/address-space class, and caller/session ownership without allocating during IR construction. |
| `CE-CCP1-F01-010` | Implement prepared stage graphs | Represent stage identity/kind, candidate, dependencies, bindings, resources, input/output order, structure epoch, value generations, global/local ranges, and profiler indices. |
| `CE-CCP1-F01-011` | Implement launch and synchronization dependencies | Represent streams as symbolic classes, events/readiness tokens, same-stream elision, cross-stream waits, transfers, device links, and host synchronization only when explicit. |
| `CE-CCP1-F01-012` | Implement generation readiness and publication | Represent preparing/ready components, pending/current generation, complete publication, no partial publication, retained order, and canonicalization request. |
| `CE-CCP1-F01-013` | Implement graph-capture and rebind contracts | Represent capture compatibility, fixed versus rebindable pointers/streams/generations, graph-stable addresses, update policy ownership, and replay variants. |
| `CE-CCP1-F01-014` | Implement symbolic runtime bindings | Represent input/output/value/workspace/native-symbol bindings by typed slots; live pointers, streams, handles, events, and leases enter only at prepared/runtime binding. |
| `CE-CCP1-F01-015` | Implement lowering-resumption checkpoints | Map canonical source, atom evidence, semantic atom, target cover, physical projection, packed operand, executable recipe, and local realization artifacts to CEIR facets with exact invalidation rules. |
| `CE-CCP1-F01-016` | Implement Realization IR text/parser/printer | Add compact syntax for targets, covers, planes, projections, packing, stages, readiness, bindings, and native fragments with unknown extension preservation. |
| `CE-CCP1-F01-017` | Implement Realization IR validators and referees | Separate structural parseability, semantic consistency, exact coverage, resource/capability checks, and unsafe continuation. |
| `CE-CCP1-F01-018` | Deliver selected-plan-to-prepared-program slice | Lower a selected Planning IR relation plan into writable Realization IR, prepare existing program_v2-compatible stages, bind runtime operands, and execute through a reference backend. |

### F02: backend ABI and CPU

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-F02-001` | Freeze the backend provider ABI | Define target discovery, capability query, Realization IR admissibility, code emission, object production, diagnostics, toolchain identity, and optional native-fragment protocols. |
| `CE-CCP1-F02-002` | Implement backend registry and selection | Register built-in and external backends through source-linked fragments, match target/capability/toolchain constraints, and honor explicit force/fallback policy. |
| `CE-CCP1-F02-003` | Implement backend code-generation plans | Translate Realization IR into explicit generated files, embedded data, compile jobs, link jobs, support libraries, and source maps before invoking any toolchain. |
| `CE-CCP1-F02-004` | Implement generated C++ representation | Emit readable modern C++ for host stubs, CPU kernels, static data, runtime bindings, stage graphs, and calls into libCellerator where profitable. |
| `CE-CCP1-F02-005` | Implement generic CPU relation apply | Lower typed relation apply to exact CPU loops or library calls using canonical/projection order and numeric policy, with deterministic reference behavior. |
| `CE-CCP1-F02-006` | Implement CPU transpose and contraction | Lower transpose, logical-edge gradients, support contraction, and partial merges with explicit order maps and accumulation types. |
| `CE-CCP1-F02-007` | Implement CPU segment, gate, update, bundle, and chain paths | Provide complete conventional CPU fallbacks for core non-relation operation families needed by language conformance. |
| `CE-CCP1-F02-008` | Implement CPU projection, packing, and order transforms | Lower selected physical projections and explicit pack/gather/scatter/canonicalization stages while retaining conventional unpacked fallback. |
| `CE-CCP1-F02-009` | Compile generated C++ into ordinary objects | Invoke discovered GCC or Clang with tracked source maps, depfiles, ABI flags, support libraries, and reproducible output paths. |
| `CE-CCP1-F02-010` | Implement host runtime binding ABI | Bind typed operand slots, workspace, generated constants, prepared stages, and error returns through libCellerator without requiring Cellerator source at runtime. |
| `CE-CCP1-F02-011` | Map backend diagnostics to source and CEIR | Translate generated-source compiler errors and warnings through Realization/Semantic/source provenance with optional generated-code notes. |
| `CE-CCP1-F02-012` | Benchmark CPU backend complete cost | Measure compile time, generated-source size, object size, preparation, projection/packing, execution, and warm reuse against direct C++ baselines. |
| `CE-CCP1-F02-013` | Deliver the first CPU object milestone | Compile a profile-bound Cellerator relation program from `. |
| `CE-CCP1-F02-014` | Freeze the CPU/backend thin waist | Publish backend ABI, registry, generated C++ contract, CPU fallbacks, object emission, and diagnostics for downstream backends. |

### F03: NVCC

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-F03-001` | Freeze the NVCC backend contract | State that NVCC never parses Cellerator syntax; it receives generated CUDA/C++ plus explicit device/host/link jobs, target architectures, support libraries, and source maps. |
| `CE-CCP1-F03-002` | Implement CUDA source emission | Emit typed kernels, device helpers, host stubs, constants, projection views, stage launchers, and runtime bindings from Realization IR with deterministic formatting. |
| `CE-CCP1-F03-003` | Bind existing Cellerator CUDA providers | Map candidate/provider identities and prepared-state contracts to source-linked existing sm70 and generic implementations rather than regenerating every kernel. |
| `CE-CCP1-F03-004` | Generate custom relation kernels where selected | Lower exact cover, projection, numeric tuple, width, order, partial algebra, and epilogue into specialized CUDA code when no prelinked provider is chosen. |
| `CE-CCP1-F03-005` | Generate non-relation CUDA operations | Lower transpose, contraction, segment, normalization, gates, sparse updates, bundles, chains, moments, exchange, and publication stages required by selected plans. |
| `CE-CCP1-F03-006` | Implement NVCC option and architecture mapping | Translate target classes, real/virtual architecture sets, host compiler, language mode, optimization, debug/line info, RDC, libraries, and user overrides deterministically. |
| `CE-CCP1-F03-007` | Implement host/device split compilation | Support whole `. |
| `CE-CCP1-F03-008` | Implement relocatable device code and device linking | Model device symbols, libraries, nvlink steps, registration objects, and multi-translation-unit device code as explicit driver actions. |
| `CE-CCP1-F03-009` | Implement PTX, cubin, and fatbinary intermediates | Capture requested backend artifacts, embed/select them where appropriate, and record toolchain/architecture identity for resumption and diagnostics. |
| `CE-CCP1-F03-010` | Integrate CUDA runtime and library linkage | Select cudart, driver API, cuSPARSE, cuBLAS, CUB, NCCL, and Cellerator provider/runtime libraries only when required by the selected realization. |
| `CE-CCP1-F03-011` | Map CUDA diagnostics and line information | Use generated `#line`, source maps, ptxas diagnostics, resource-usage reports, and keep-temps to trace source/IR to CUDA/PTX. |
| `CE-CCP1-F03-012` | Integrate asynchronous readiness and streams | Lower symbolic stage dependencies to caller streams, events, same-stream elision, generation publication, and graph-compatible launch behavior using existing runtime contracts. |
| `CE-CCP1-F03-013` | Deliver the first NVCC object milestone | Compile a profile-bound relation program through data-aware Planning IR and Realization IR into an ordinary CUDA-capable object with NVCC, link it, and execute on sm70. |
| `CE-CCP1-F03-014` | Benchmark NVCC backend complete cost | Measure Cellerator compile/planning, generated source, nvcc time, ptxas resources, object size, preparation, packing, transfers, launches, kernel time, and reuse. |
| `CE-CCP1-F03-015` | Freeze the NVCC backend | Publish source emission, provider binding, action graph, object/fatbinary behavior, diagnostics, readiness, and performance receipts. |

### F04: Clang CUDA, NVPTX, direct PTX

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-F04-001` | Freeze Clang CUDA and NVPTX backend contracts | Define optional routes from Realization IR through generated CUDA parsed by Clang or through LLVM/NVPTX lowering, sharing the backend ABI and source maps. |
| `CE-CCP1-F04-002` | Implement Clang CUDA action mapping | Translate host/device compilation, target architecture, libdevice, CUDA includes/libraries, offload bundling, and linker steps to the discovered Clang toolchain. |
| `CE-CCP1-F04-003` | Implement LLVM/NVPTX module boundary | Lower a constrained Realization IR subset to LLVM IR/NVPTX through an internal adapter while keeping LLVM types out of public Cellerator IR APIs. |
| `CE-CCP1-F04-004` | Define direct PTX typed operation model | Represent PTX types, registers, address spaces, predicates, labels, parameters, memory effects, barriers, collectives, and instruction requirements as Realization IR extension nodes. |
| `CE-CCP1-F04-005` | Implement inline PTX/native block binding | Bind typed inputs, outputs, clobbers, memory/order/effect contracts, target predicates, fallback, and source provenance for expert inline blocks. |
| `CE-CCP1-F04-006` | Implement PTX emission and ptxas assembly | Emit deterministic PTX, invoke ptxas when available, capture cubin/resource diagnostics, and support retained PTX for driver-JIT loading without defining the full Part Two JIT system. |
| `CE-CCP1-F04-007` | Implement fatbinary/object embedding for direct PTX | Package PTX/cubin plus registration/launch stubs into ordinary linkable objects through supported toolchain mechanisms. |
| `CE-CCP1-F04-008` | Map target capabilities and instruction families | Validate compute capability, WMMA/MMA shapes, collective scope, layouts, sparsity, memory interfaces, and numeric tuples against current capability manifests. |
| `CE-CCP1-F04-009` | Implement source-to-PTX provenance | Map source field/operation, Semantic/Planning/Realization nodes, generated PTX lines, and assembled resource reports through cold sidecars. |
| `CE-CCP1-F04-010` | Deliver a direct-PTX hot-path demonstration | Lower one narrowly scoped, exact, target-specific relation or packing microkernel directly to PTX and execute it without NVCC parsing generated device code. |
| `CE-CCP1-F04-011` | Compare NVCC, Clang CUDA, and direct PTX routes | Measure compile time, object size, resource usage, launch/execution, diagnostics, and maintainability for matched realizations. |
| `CE-CCP1-F04-012` | Validate backend fallback and mixed routes | Allow one program to use prelinked providers, generated NVCC code, and direct PTX stages when ABI/order/generation contracts permit; fall back cleanly when an optional route is unavailable. |
| `CE-CCP1-F04-013` | Freeze optional NVIDIA backend routes | Publish supported subsets, capability contracts, diagnostics, provenance, and evaluated promotion status without making LLVM or direct PTX mandatory for host/NVCC builds. |
