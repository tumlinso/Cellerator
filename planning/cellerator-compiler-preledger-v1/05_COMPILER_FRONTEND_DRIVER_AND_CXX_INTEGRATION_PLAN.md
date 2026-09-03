# Compiler frontend, driver, and C++ integration plan

## Architectural result

Cellerator is its own compiler driver and semantic frontend. It is not a permanent Clang fork.

The frontend uses upstream Clang libraries behind a versioned adapter for the parts of C++ that are too large and too subtle to duplicate: preprocessing integration, lookup, templates, overload resolution, concepts, constant evaluation, target types, and tooling.

Cellerator owns:

- file-local dialect activation;
- Cellerator token recognition and parsing;
- source-to-shadow mappings;
- Cellerator AST and Sema;
- CEIR construction;
- profiles, planning, realization, and backend actions.

The selected downstream toolchain owns conventional object generation and linking unless Cellerator directly emits a lower artifact such as PTX.

## Source pipeline

```text
physical source
    -> ordinary preprocessing with FileID-scoped #pragma state
    -> lossless activated token islands
    -> Cellerator parse tree
    -> hygienic shadow C++ placeholders
    -> upstream Clang parsing/Sema of C++ captures
    -> resolved C++ types/templates/constants/effects
    -> Cellerator AST and biological Sema
```

The shadow C++ is an internal semantic device, not the public language and not a macro implementation.

Cellerator syntax produced by macro expansion is parsed after preprocessing. Provenance retains both definition and expansion locations. Activation is determined by the active physical include instance.

## Driver behavior

The driver supports preprocess, syntax-only, CEIR emission, compile-only, assemble, link, inspect, profile, and LTO actions.

Toolchain discovery follows deterministic precedence:

1. explicit command-line override;
2. response-file setting;
3. dedicated environment variable;
4. configured/installed compiler resource manifest;
5. PATH and platform discovery.

The driver records a toolchain fingerprint used by artifacts, caches, and lowering resumption.

When no activated Cellerator syntax exists, the driver builds no biological IR and falls through to the selected C++ compiler.

## Build independence

The frontend and driver build without CUDA. CUDA routes appear as optional backend providers. `CELLERATOR_ENABLE_CUDA=AUTO` preserves accelerator-first behavior on capable systems without making NVCC a root configure prerequisite.

## Workstream task catalog

### B01: host-only build partition

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-B01-001` | Define the host-only root project contract | Change the planned root-language contract from unconditional CXX+CUDA to CXX with explicit optional accelerator enablement. |
| `CE-CCP1-B01-002` | Define tri-state CUDA backend configuration | Specify CELLERATOR_ENABLE_CUDA=AUTO|ON|OFF, check_language(CUDA), explicit failure for ON without a toolchain, and non-failure for AUTO/OFF. |
| `CE-CCP1-B01-003` | Create compiler component target boundaries | Define Cellerator::CompilerCore, Frontend, CEIR, Profiles, Planning, Realization, Backends, Tooling, and Diagnostics targets with acyclic link directions. |
| `CE-CCP1-B01-004` | Set modern compiler/tooling language standards | Build compiler and tooling implementation with a modern C++ baseline, initially C++23 where supported, while retaining explicit C++17/CUDA17 compatibility islands for existing runtime/provider code. |
| `CE-CCP1-B01-005` | Isolate legacy CUDA target requirements | Move CUDA language properties, CUDAToolkit discovery, architecture flags, and provider manifests behind CUDA-enabled target functions rather than root-global requirements. |
| `CE-CCP1-B01-006` | Define optional LLVM and Clang library discovery | Discover compatible upstream LLVM/Clang development packages for frontend integration without pinning a fork. |
| `CE-CCP1-B01-007` | Define backend resource discovery manifests | Generate a cold compiler resource manifest naming available host compilers, nvcc, clang CUDA, LLVM/NVPTX, ptxas, linkers, and support directories without probing devices in hot paths. |
| `CE-CCP1-B01-008` | Define build-tree generated header ownership | Generate compiler version, language revision, CEIR revision, backend capability, and install-resource path headers into the build tree. |
| `CE-CCP1-B01-009` | Create host-only compiler smoke targets | Add planned smoke targets for CEIR parser/printer, profile loader, source manager, diagnostics, and celleratord protocol code without CUDA linkage. |
| `CE-CCP1-B01-010` | Create accelerator-enabled compiler smoke targets | Add conditional linkage tests that bind Realization IR and backend adapters to existing Cellerator CUDA provider/runtime targets. |
| `CE-CCP1-B01-011` | Define build presets and CI matrix | Specify presets for host-only Clang, host-only GCC, CUDA+NVCC sm70, CUDA+Clang where available, installed-consumer, sanitizer, and language-server builds. |
| `CE-CCP1-B01-012` | Freeze the compiler target graph | Publish the target dependency graph, feature options, standards, generated headers, and configure behavior as the build thin waist. |

### B02: compiler driver

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-B02-001` | Define the compiler invocation and action graph | Model preprocess, analyze, emit CEIR, compile, assemble, device-link, host-link, and inspect actions as explicit driver jobs with stable diagnostics and no backend-specific assumptions in semantic stages. |
| `CE-CCP1-B02-002` | Implement response-file and argv normalization contracts | Specify deterministic response-file expansion, quoting, environment capture, path normalization, and forwarding groups so very large downstream command lines remain reproducible. |
| `CE-CCP1-B02-003` | Discover host Clang toolchains | Search explicit overrides, environment, configured resources, PATH, and platform defaults in a documented precedence order. |
| `CE-CCP1-B02-004` | Discover host GCC toolchains | Resolve g++, gcc, linker, include search, target triple, libstdc++ ABI mode, and version identity independently of the Clang semantic-library dependency. |
| `CE-CCP1-B02-005` | Discover NVCC toolchains | Resolve nvcc, host compiler compatibility, CUDA toolkit root, ptxas, nvlink, fatbinary, architecture support, and version identity with explicit overrides. |
| `CE-CCP1-B02-006` | Discover Clang CUDA and LLVM/NVPTX toolchains | Resolve clang++, LLVM libraries/tools, CUDA resource paths, libdevice, target support, and ptxas availability without requiring these routes to exist. |
| `CE-CCP1-B02-007` | Define toolchain override precedence | Support command-line, response file, environment, build configuration, installed resource manifest, and system discovery in one deterministic policy analogous to nvcc. |
| `CE-CCP1-B02-008` | Track downstream C++ language and ABI mode | Parse and forward -std, target, exception, RTTI, sanitizer, visibility, ABI, include, macro, and linker flags while separately selecting the compiler's own implementation standard. |
| `CE-CCP1-B02-009` | Implement plain C++ passthrough planning | When no activated Cellerator syntax exists, construct a zero-semantic-work driver plan that invokes the downstream compiler with equivalent arguments and exit behavior. |
| `CE-CCP1-B02-010` | Define compilation database and dependency-file behavior | Emit or preserve compile_commands entries, depfiles, module dependencies, and source-to-output mappings suitable for CMake, Ninja, and celleratord. |
| `CE-CCP1-B02-011` | Define temporary artifact and cache policy | Use deterministic per-action temporary directories, explicit keep-temps modes, content-addressed cold caches, and cleanup that never hides artifacts requested for diagnostics. |
| `CE-CCP1-B02-012` | Forward and remap downstream diagnostics | Preserve severity, source ranges, fix-its, and exit codes from downstream compilers while allowing Cellerator source maps to remap generated/shadow locations. |
| `CE-CCP1-B02-013` | Fingerprint toolchains for artifacts and resumption | Create a stable toolchain identity from executable content/version, target, resource directory, critical flags, runtime/driver identity, and backend plugin revision. |
| `CE-CCP1-B02-014` | Deliver the driver passthrough milestone | Build bin/cellerator as a thin main over the shared driver library and compile/link ordinary C++ through Clang and GCC without Cellerator semantics. |

### B03: source manager and shadow C++

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-B03-001` | Define the unified source-location model | Represent physical files, include instances, macro expansions, transformed buffers, CEIR nodes, and backend output with stable source spans and reversible mapping edges. |
| `CE-CCP1-B03-002` | Classify source inputs independently of extension | Treat . |
| `CE-CCP1-B03-003` | Register the #pragma cellerator preprocessor contract | Recognize the pragma through an upstream-compatible preprocessor hook, record activation source location and revision options, and diagnose malformed forms. |
| `CE-CCP1-B03-004` | Implement file-local dialect state | Associate activation with the current physical FileID/include instance so mode begins at the pragma, ends at that file boundary, and never leaks to includers or included files. |
| `CE-CCP1-B03-005` | Define pragma interaction with preprocessing conditionals | Specify activation inside #if branches, inactive directives, include replay, precompiled headers, and modules. |
| `CE-CCP1-B03-006` | Build a lossless raw-token stream | Capture spelling, trivia, source span, macro origin, activated dialect state, and preprocessor condition for every token needed by Cellerator parsing and source reconstruction. |
| `CE-CCP1-B03-007` | Define macro expansion semantics for Cellerator tokens | Parse Cellerator constructs after preprocessing while preserving definition and expansion provenance. |
| `CE-CCP1-B03-008` | Recognize Cellerator execution-field token islands | Detect balanced <[ . |
| `CE-CCP1-B03-009` | Recognize relation and operation token forms | Identify -[relation]-> and other language-spec operation families without assigning semantics during the raw source pass. |
| `CE-CCP1-B03-010` | Construct shadow C++ placeholders | Replace Cellerator constructs with valid generated C++ expressions/declarations carrying stable placeholder IDs, typed capture slots, and source-map anchors while retaining unaffected C++ verbatim. |
| `CE-CCP1-B03-011` | Map shadow AST nodes back to Cellerator syntax | Maintain one-to-one or explicit many-to-one mappings from placeholder declarations/calls to Cellerator parse nodes and source captures. |
| `CE-CCP1-B03-012` | Define generated identifier hygiene | Use reserved internal namespaces and content-derived identifiers that cannot collide with user macros, symbols, modules, or link names. |
| `CE-CCP1-B03-013` | Cache preprocessed activated headers safely | Key cached token/shadow products by file content, pragma revision, macro environment, include context, and frontend adapter identity. |
| `CE-CCP1-B03-014` | Expose source-pipeline diagnostics and dumps | Provide token, activation-map, shadow-source, and source-map dumps without making them required hot-path artifacts. |
| `CE-CCP1-B03-015` | Deliver the pragma-aware source milestone | Compile a mixed translation unit containing ordinary C++, activated Cellerator syntax, inactive includes, and an activated header through source transformation into a valid Clang parse. |

### B04: C++ semantic bridge

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-B04-001` | Freeze the upstream Clang adapter boundary | Hide Clang-version-specific AST, Sema, Preprocessor, diagnostics, and tooling APIs behind a versioned Cellerator adapter so the public compiler API does not expose unstable Clang internals. |
| `CE-CCP1-B04-002` | Create the C++ compilation invocation bridge | Construct Clang CompilerInvocation state from normalized cellerator driver arguments, target, sysroot, includes, macros, modules, and language mode. |
| `CE-CCP1-B04-003` | Parse shadow translation units with full C++ semantics | Run preprocessing, parsing, lookup, overload resolution, template instantiation, constexpr evaluation, and diagnostics over the generated shadow source. |
| `CE-CCP1-B04-004` | Bind source captures to C++ declarations and expressions | Resolve every Cellerator domain, state, relation, qualifier expression, native call, and inline-IR capture to a typed Clang AST handle plus source provenance. |
| `CE-CCP1-B04-005` | Extract canonical and spelled C++ types | Preserve both canonical type identity for planning and user spelling for diagnostics, including __half, bf16 wrappers, vectors, pointers, references, address spaces, and user-defined numeric types. |
| `CE-CCP1-B04-006` | Integrate template instantiation with typed biological operations | Instantiate source templates before final operation selection while retaining dependent biological constraints until substitution resolves numeric and domain types. |
| `CE-CCP1-B04-007` | Integrate overload resolution and Cellerator semantic candidates | Allow C++ overloads/concepts to choose declarations while Cellerator Sema validates biological compatibility and operation semantics after C++ resolution. |
| `CE-CCP1-B04-008` | Expose constexpr and constant-evaluation results | Import compile-time extents, policies, profile names, reuse counts, and user constants into Cellerator Sema without duplicating a C++ evaluator. |
| `CE-CCP1-B04-009` | Model opaque native calls | Treat unresolved or uncontracted C/C++ calls inside fields as explicit semantic barriers with conservative read/write/escape effects, not as invisible operations. |
| `CE-CCP1-B04-010` | Bind native effect contracts | Associate source-level contracts with resolved functions and validate reads, writes, topology/order/support/value effects, determinism, purity, and aliasing claims. |
| `CE-CCP1-B04-011` | Reconcile GCC-hosted ABI and library semantics | When GCC is the downstream compiler, compare Clang-derived semantic assumptions with GCC target, standard library, ABI macros, and calling conventions; diagnose unsupported mismatches. |
| `CE-CCP1-B04-012` | Preserve pure C++ fallthrough exactly | Skip Cellerator AST/IR construction for unactivated source and preserve compilation, diagnostics, depfiles, modules, and link semantics through the driver. |
| `CE-CCP1-B04-013` | Expose reusable frontend sessions | Provide cancellable, thread-aware parse/Sema sessions and immutable snapshots suitable for libCellerator and celleratord rather than coupling the bridge to one command-line process. |
| `CE-CCP1-B04-014` | Freeze the C++ semantic bridge milestone | Publish the versioned adapter and demonstrate an activated Cellerator placeholder resolving real C++ names, templates, constexpr values, and numeric types. |
