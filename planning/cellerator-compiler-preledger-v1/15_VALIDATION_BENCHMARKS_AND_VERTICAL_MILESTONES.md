# Validation, benchmarks, and vertical milestones

## Validation strategy

Correctness pressure is independent and layered:

- source grammar and specification conformance;
- ordinary C++ differential compilation;
- biological Sema negative tests;
- CEIR text/binary fuzzing and round trip;
- profile corruption, transfer, join, and exact-summary checks;
- exact coverage and partial-algebra property tests;
- pass/extension crash, timeout, and trust-mode tests;
- cross-TU/object corruption;
- CPU and CUDA differential execution;
- sanitizers, resource bounds, cancellation, process cleanup.

## Complete-cost performance

Relevant tasks measure only the costs they materially affect, selected from:

- driver, preprocessing, Sema, AST, CEIR, profiles, discovery, planning, realization, backend and link time;
- host/device memory and allocation;
- source, CEIR, object, fatbinary, and executable size;
- preparation, projection, packing, transfer, synchronization, launch, execution, canonicalization, and reuse;
- register use, spills, occupancy, graph replay;
- celleratord startup, diagnostics, edit, completion, hover, deep query latency.

Experimental work may finish evaluated-not-promoted.

## Progressive vertical milestones

| Milestone | Demonstration |
| --- | --- |
| M10 | `cellerator` passes ordinary C++ through GCC/Clang and understands file-local pragma/token islands |
| M20 | real Cellerator grammar, AST, biological Sema, fields, effects, and profile-required semantic slice |
| M30 | source -> writable Semantic IR plus multi-state representative profiles |
| M40 | inspectable Planning IR with rehomed JBC discovery, exact atoms, grammar/basis, decomposition, candidates, and complete costs |
| M50 | writable Realization IR plus ordinary CPU object and sm70 NVCC object |
| M60 | reflection, inline IR, custom/replacement passes, bounded same-compilation transforms, unsafe control, provenance |
| M70 | cross-TU CEIR/LTO, libCellerator, `.cell` stdlib, reference profiles, relocatable SDK |
| M80 | celleratord ordinary C++ parity plus Cellerator semantic/planning/native queries |
| M90 | clean host and NVIDIA final acceptance, docs reconciled, JBC preserved, Part Two separated |

## Workstream task catalog

### J01: conformance and resilience

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-J01-001` | Build the source-language conformance corpus | Turn every normative syntax/semantic example and rejected design into positive/negative tests with specification section references. |
| `CE-CCP1-J01-002` | Fuzz activated-source lexing and pragma state | Generate C++ token streams, includes, macros, templates, comments, and Cellerator delimiters to find activation leaks, hangs, and source-map corruption. |
| `CE-CCP1-J01-003` | Fuzz parser and recovery | Mutate declarations, fields, relation syntax, controls, inline IR, reflection, passes, and native fragments while enforcing bounded diagnostics and no crashes. |
| `CE-CCP1-J01-004` | Differentially test ordinary C++ fallthrough | Compile broad C++ corpora through cellerator and direct Clang/GCC, comparing success, diagnostics categories, object symbols, depfiles, and runtime output. |
| `CE-CCP1-J01-005` | Build biological Sema negative suites | Cover shape-only equivalence, wrong endpoints/orders/generations/orientation, stale values, invalid effects, unsupported numeric tuples, and unsafe override behavior. |
| `CE-CCP1-J01-006` | Fuzz CEIR text and binary artifacts | Mutate all three IR levels, unknown extensions, identities, regions, use-def chains, native payloads, directories, hashes, and versions. |
| `CE-CCP1-J01-007` | Fuzz profile artifacts and state propagation | Corrupt profile sections/evidence, generate branch/state graphs, and compare transfer/join/widen results with small exact models. |
| `CE-CCP1-J01-008` | Property-test exact coverage and partial algebra | Generate decompositions, halos, replicas, contributors, merge trees, and canonical maps; independently prove reconstruction or detect error. |
| `CE-CCP1-J01-009` | Test pass and extension isolation | Inject invalid IR, unknown extensions, crashes, timeouts, recursive self-transforms, stale plugins, and false preservation claims in every trust mode. |
| `CE-CCP1-J01-010` | Test cross-TU/LTO and object corruption | Mutate embedded/sidecar CEIR, duplicate symbols, conflicting profiles, archives, stripped sections, and mixed plain objects. |
| `CE-CCP1-J01-011` | Run sanitizers and resource-bound tests | Exercise ASan, UBSan, TSan where supported, leak checks, file descriptor/process cleanup, timeout, memory capacity, and candidate explosion controls. |
| `CE-CCP1-J01-012` | Freeze independent conformance and resilience acceptance | Aggregate source, CEIR, profile, planner, realization, backend, LTO, extension, SDK, and celleratord test receipts with no unexplained regressions. |

### J02: performance and milestones

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-J02-001` | Freeze compiler benchmark methodology | Define hardware/toolchain/profile/source identities, cold/warm modes, repetitions/statistics, contamination, peak memory, generated artifact sizes, and exact baselines. |
| `CE-CCP1-J02-002` | Benchmark plain C++ passthrough | Measure driver overhead, preprocess/compile/link wall time, peak RSS, depfiles, object size, and diagnostics against direct Clang/GCC. |
| `CE-CCP1-J02-003` | Benchmark source frontend and C++ bridge | Measure preprocessing, activated-token analysis, shadow generation, Clang Sema, AST construction, incremental reuse, and source-map memory. |
| `CE-CCP1-J02-004` | Benchmark CEIR construction and serialization | Measure node construction, canonicalization, text parse/print, binary load/store, memory mapping, unknown extensions, and provenance stripping across all levels. |
| `CE-CCP1-J02-005` | Benchmark representative profiles | Measure profile build, exact scans, sketches, load, mapped queries, state transfer, branch joins, multi-state alternatives, and memory. |
| `CE-CCP1-J02-006` | Benchmark discovery and certification | Measure each migrated JBC proposal mechanism, candidate counts, exact rescans, certification, memory, no-basis cases, and matched generic/null baselines. |
| `CE-CCP1-J02-007` | Benchmark planning and complete costs | Measure decomposition portfolios, candidate enumeration, transition costs, planner quality versus oracle, external costs, cache/resumption, and profile variants. |
| `CE-CCP1-J02-008` | Benchmark Realization IR and backend generation | Measure realization, projection/packing planning, stage construction, generated source, downstream compiler time, ptxas resources, object/fatbinary size, and provenance. |
| `CE-CCP1-J02-009` | Benchmark generated CPU execution | Measure preparation, transforms, packing, execution, reuse, memory, and output recovery against direct C++ and existing Cellerator runtime paths. |
| `CE-CCP1-J02-010` | Benchmark generated NVIDIA execution | Measure preparation, allocation, packing, transfer, synchronization, launches, kernel time, canonicalization, graph replay, and amortized reuse on sm70. |
| `CE-CCP1-J02-011` | Benchmark cross-TU/LTO | Measure object CEIR size, extraction, merge, incremental cache, replanning, re-emission, link time, binary size, and runtime benefit for authorized field chains. |
| `CE-CCP1-J02-012` | Benchmark libCellerator and celleratord | Measure API session startup, concurrent parses, cancellation, editor startup, diagnostics, completion, hover, IR/candidate queries, and memory. |
| `CE-CCP1-J02-013` | Execute progressive vertical milestone campaign | Record driver passthrough, pragma parse, first Semantic IR, profile-aware compile, Planning IR candidate, CPU object, NVCC object, inline rewrite, custom pass, cross-TU import, and celleratord hover as separately reproducible milestones. |
| `CE-CCP1-J02-014` | Freeze the Part One performance and milestone receipt | Aggregate complete-cost evidence, regression budgets, promotion/non-promotion decisions, source/toolchain/profile identities, and milestone reproducibility. |
