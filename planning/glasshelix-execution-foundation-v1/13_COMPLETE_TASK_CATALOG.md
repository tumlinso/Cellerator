# Complete proposed task catalog

Cellerator: 79 leaf tasks plus one epic. All are planned on delivery.

## A: Live baseline, preserved work and runtime qualification

### CE-NF1-A01: Revalidate source and both planning precedents
Inspect current AGENTS, scope, operation-core v2, JBC transition map, SS1 and RU1 source/tests; classify implemented, unlinked, reference-only and absent behavior before changing it.

Acceptance: Record actual HEAD, worktrees, dirty files, task authority and callable target map.
Failure to reject: Existing dirty work and unrelated active runs remain unchanged; old completion labels never substitute for code inspection.
Dependencies: none

### CE-NF1-A02: Resolve first-class dispatch and host evidence services
Discover the installed authoritative claim, lane, dispatch, rendezvous, workspace, patch and integration front doors; bind the two repository identities under the GlassHelix controller. Discover hardware lease and performance serialization services.

Acceptance: A machine-readable execution binding records verified command/tool contracts without starting work as part of bootstrap.
Failure to reject: A local child process cannot be counted as a first-class lane; unresolved dispatch or lease capability remains an explicit blocker.
Dependencies: CE-NF1-A01

### CE-NF1-A03: Inventory native consumer reachability and numerical gaps
Trace external linking of prepared programs, value planes, gates, segment kernels and relation updates; identify actual CPU and CUDA routes and inherited N1/N16 and f16 restrictions.

Acceptance: Capability inventory separates schema declarations, implementations, linked targets and executed evidence.
Failure to reject: No general-width, f32-storage, JVP, n-ary or mutable-capture claim is inferred from a header name.
Dependencies: CE-NF1-A02

### CE-NF1-A04: Adopt constructive extension and regression boundary
Record disposition for useful algorithms, legacy consumers, existing generation protocols and deferred compiler work; assign central-file ownership and a narrowly scoped native foundation target.

Acceptance: Regression obligations include the retained relation update and existing consumers affected by edits.
Failure to reject: No reset, broad cleanup, old-run hijack, CE-AMP activation or second numerical runtime is authorized.
Dependencies: CE-NF1-A03

## C: Numerical contracts and capabilities

### CE-NF1-C01: Define multi-operand operation and effect contract
Extend the existing operation family with explicit argument/output roles, domain/order identities, counts, non-aliasing or permitted aliasing, output initialization and numerical semantics; keep provider details outside scientific identity.

Acceptance: A three-input two-output operation can be described without flattening it into a fake pairwise relation.
Failure to reject: Wrong order, wrong domain, duplicate output ownership and unknown capabilities are rejected before launch.
Dependencies: CE-NF1-M00

### CE-NF1-C02: Define immutable preparation and independent bindings
Freeze ownership of immutable support/projections separately from mutable parameters, state and gradients; distinguish definition reuse, structural reuse, parameter tying and output memoization.

Acceptance: Two value instances can legally bind one prepared structure with independent generations.
Failure to reject: Equal pointers, equal initial values or common code do not imply parameter tying or historical snapshots.
Dependencies: CE-NF1-C01

### CE-NF1-C03: Define derivative and differential provenance contract
Describe optional JVP, VJP and supported second-direction actions, domains, primal generations, forward branch decisions and differentiation conventions around rounding.

Acceptance: Missing derivative is a capability failure, never a numerical zero.
Failure to reject: Stale values, foreign state generations and unsupported nonsmooth points cannot silently enter derivative results.
Dependencies: CE-NF1-C02

### CE-NF1-C04: Define support activity, epochs and accuracy contract
Distinguish structural membership, exact predicate exclusion, numerical zero, compact active projection and approximate dropping; define separate primal and response support requirements.

Acceptance: Value, mask and structural changes have separately inspectable invalidation effects.
Failure to reject: A zero primal contribution does not automatically delete a trainable response path; nonfinite masking semantics remain explicit.
Dependencies: CE-NF1-C03

### CE-NF1-C05: Define custom compiled block contract
Support native precompiled operations with declared dependencies, side effects, arithmetic and optional derivatives; use grouped evaluator dispatch rather than per-edge virtual calls.

Acceptance: A consumer-defined nonlinear block joins the existing program path without teaching GlassHelix a new tensor engine.
Failure to reject: Unknown effects conservatively prevent fusion/reuse; no fake generic JIT or automatic derivative is promised.
Dependencies: CE-NF1-C04

### CE-NF1-C06: Freeze developmental native capability seam
Publish versioned capability and ownership documents with source/interface hashes for downstream import, allowing coordinated revision rather than permanent ABI freezing.

Acceptance: Each required capability has one owner, concrete acceptance and honest unsupported behavior.
Failure to reject: Contract changes revise the receipt and dependent scopes rather than silently breaking consumers.
Dependencies: CE-NF1-C05

## B: Focused build and separate native consumer

### CE-NF1-B01: Partition host semantic and CUDA target dependencies
Expose the selected native foundation through source-linked CMake targets; host contracts and CPU reference execution must not require CUDA toolkit discovery.

Acceptance: Clean host-only configure/build links a standalone consumer.
Failure to reject: CUDA OFF cannot transitively invoke nvcc or require Torch, CellShard or model targets.
Dependencies: CE-NF1-M10

### CE-NF1-B02: Expose existing reusable operation targets
Reuse prepared relation, segment, gate, value-plane and program implementations through owned targets rather than including implementation files in consumers.

Acceptance: External consumer links the actual library implementations through declared targets.
Failure to reject: A demo that includes a private .cu implementation is rejected as boundary validation.
Dependencies: CE-NF1-B01

### CE-NF1-B03: Make toolchain and architecture selection explicit
Respect caller compilers, CUDA architecture and build directory; avoid machine-local hard-coded paths or setting CXX compiler after project initialization.

Acceptance: Record CUDA 12.x/sm70 qualified configuration and host C++20 configuration separately.
Failure to reject: Do not require CUDA 13 or claim FP8 arithmetic on V100; unavailable toolchains fail with actionable diagnostics.
Dependencies: CE-NF1-B02

### CE-NF1-B04: Gate optional suites and provide CPU correctness target
Stop optional test/example/legacy dependencies from leaking into minimal consumer builds while preserving their opt-in configurations.

Acceptance: Minimal build and regression build both configure independently.
Failure to reject: Hiding or removing existing regressions to make minimal build green is not accepted.
Dependencies: CE-NF1-B03

### CE-NF1-B05: Add relocatable build-tree consumer proof
Supply explicit exported or build-tree targets, include propagation and dependency revision manifest; no installed SDK or .cell compiler required.

Acceptance: A fresh directory compiles and links a consumer using only supported includes and targets.
Failure to reject: A same-translation-unit smoke test cannot satisfy external linking.
Dependencies: CE-NF1-B04

### CE-NF1-B06: Qualify build matrix and publish target receipt
Execute host-only and V100 build matrix, independently verify no forbidden dependencies, and publish exact target/capability receipt.

Acceptance: Target receipt names concrete libraries, build identities and successful commands.
Failure to reject: Stale builds, skipped targets and header-only stubs cannot satisfy the receipt.
Dependencies: CE-NF1-B05

## P: Prepared program and launch execution

### CE-NF1-P01: Generalize stage bindings without a second runner
Extend the existing prepared program to accept typed operand/value collections and effects instead of one opaque input/output pair.

Acceptance: Multi-input and multi-output stages execute through the canonical runner.
Failure to reject: The legacy runner stays covered; GlassHelix cannot acquire parallel scheduler ownership.
Dependencies: CE-NF1-M10

### CE-NF1-P02: Preflight all dynamic launch bindings
Validate every stage binding, capacity, axis, workspace and alias requirement before submitting the first stage; cache only static validation.

Acceptance: Invalid final-stage bindings leave all outputs and enqueue counters untouched.
Failure to reject: Per-stage late checking after prior effects is a regression.
Dependencies: CE-NF1-P01

### CE-NF1-P03: Bind sessions and independently reusable instances
Connect prepared program ownership to existing execution_session scratch and stream resources; use one owner stream per instance initially.

Acceptance: Two serially driven instances share immutable program data without sharing mutable scratch.
Failure to reject: Host serialization assumptions, device ownership and external read boundaries are enforced rather than implied.
Dependencies: CE-NF1-P02, CE-NF1-V03

### CE-NF1-P04: Plan scratch lifetimes and bounded state buffers
Compute whole-program scratch requirements, liveness reuse and state ping-pong buffers on preparation; preserve borrowed buffer lifetime and checked teardown.

Acceptance: Repeated steady-state execution has no hidden allocations or whole-state canonicalization.
Failure to reject: Alias overlap and use-after-completion assumptions are tested with independent sentinels.
Dependencies: CE-NF1-P03

### CE-NF1-P05: Implement failure and poison semantics
Separate preflight rejection from failures after partial submission; record accepted enqueue versus observed completion; propagate device failure to invalid dependent results.

Acceptance: Injected late submission failure cannot produce a valid generation or reusable result.
Failure to reject: No rollback claim is made after in-place numeric effects have occurred.
Dependencies: CE-NF1-P04

### CE-NF1-P06: Map stage provenance and counters
Retain scientific-operation identifiers through decomposition/fusion, source/value generations and selected candidate identifiers with low-overhead counters.

Acceptance: Fused and unfused executions report the same originating numerical operations.
Failure to reject: Stage IDs are not presented as biological actors or evidence of molecular identity.
Dependencies: CE-NF1-P05

### CE-NF1-P07: Qualify linked composed-program boundary
Run lifecycle, multi-operand, ordering, failure and reuse cases from an external native consumer; publish program capability evidence.

Acceptance: Program execution is real on host and V100, with late-invalid-input no-side-effects proof.
Failure to reject: A metadata-only stage-graph validator cannot satisfy composed execution.
Dependencies: CE-NF1-P06, CE-NF1-B05, CE-NF1-N05

## V: Shared structure, value instances and readiness

### CE-NF1-V01: Extract reusable structural preparation ownership
Retain actual forward/transpose algorithms while separating immutable topology/projections from the single mutable value plane in the bounded pair.

Acceptance: Two independently owned value instances share one counted structural preparation.
Failure to reject: No duplicate topology storage is hidden behind a wrapper; retained pair behavior remains compatible.
Dependencies: CE-NF1-M10

### CE-NF1-V02: Bind value and gradient instances through current planes
Connect projection_value_plane and atom-plane contracts to executable instances with explicit logical or physical-primary ownership.

Acceptance: Values can be rebound at a new address/generation without changing topology identity.
Failure to reject: Physical padding holes never become trainable parameters or valid logical edges.
Dependencies: CE-NF1-V01

### CE-NF1-V03: Preserve asynchronous publication and read completion
Extend existing owner-stream readiness semantics to instances; readiness publication records enqueue, not completion; retain existing limited reader protocol where applicable.

Acceptance: Delayed reader completion prevents unsafe in-place mutation and checked close refuses live borrows.
Failure to reject: One owner cannot mutate another instance or consume an unreturned external lease.
Dependencies: CE-NF1-V02

### CE-NF1-V04: Support f32 authoritative values alongside f16 projections
Add a true f32 relation/value path and optional explicitly derived lower-precision execution planes; no hidden rounding policy or trainable master copy.

Acceptance: Small f32 parameter changes survive publication and produce expected changes where observable.
Failure to reject: Inferred low-precision changes are not treated as exact biology; projection refresh follows the real generation.
Dependencies: CE-NF1-V03

### CE-NF1-V05: Make epoch replacement and stale-ticket rejection explicit
Publish a new structural epoch at a safe boundary, retire old executable projections only after use, and reject stale values/gradients.

Acceptance: Only affected preparation is rebuilt after a structural change.
Failure to reject: Memory reuse cannot make an old ticket valid for a new pair incarnation.
Dependencies: CE-NF1-V04

### CE-NF1-V06: Qualify value provenance under update and branching
Exercise caller deltas, parameter changes, response stamps and two mechanism instances with independent values; cover both old and new routes.

Acceptance: Updating one instance invalidates its responses and preserves the other's results.
Failure to reject: Shared structure does not imply shared parameter mutation or a historical snapshot guarantee.
Dependencies: CE-NF1-V05

### CE-NF1-V07: Publish executable shared-value capability
Independently validate structure sharing, f32 and f16 policies, readiness, close behavior and retained RU1 regressions.

Acceptance: Source-linked capability and tests identify actual supported widths and arithmetic.
Failure to reject: Unsupported stream, capture and device combinations fail before effects.
Dependencies: CE-NF1-V06

## N: General-width numeric operations and reference routes

### CE-NF1-N01: Provide independent host f64 and executable f32 references
Establish small logical-order reference operations and a native CPU execution route using explicit arithmetic, not a CUDA-required compiler adapter.

Acceptance: Independent reference detects deliberately permuted endpoints and incorrect reductions.
Failure to reject: The oracle must not reuse optimized index maps or the same lowering implementation.
Dependencies: CE-NF1-M10

### CE-NF1-N02: Generalize relation application across widths and tails
Reuse existing candidates and width decomposition, adding a correctness fallback for arbitrary positive widths and legal empty cases.

Acceptance: Widths 1, 3, 15, 16, 17, 33 and 65 agree with independent references.
Failure to reject: Padding cannot appear as biological entities or alter output normalization.
Dependencies: CE-NF1-N01, CE-NF1-V07

### CE-NF1-N03: Generalize transpose and explicit destination effects
Provide overwrite, accumulate and permitted affine effects with declared initialization; preserve source/destination order and numerical policy.

Acceptance: Forward/transpose adjoint consistency holds under matching policy and irregular support.
Failure to reject: Wrong-order outputs, uninitialized accumulations and illegal aliases fail preflight.
Dependencies: CE-NF1-N02

### CE-NF1-N04: Provide batched local arithmetic and gather/scatter
Add missing reusable arithmetic, argument gathering and deterministic destination assembly via existing operations where possible.

Acceptance: Nonlinear program stages can remain in prepared order without roundtripping through canonical arrays.
Failure to reject: Duplicate destination writes require an explicit reduction, never a data race.
Dependencies: CE-NF1-N03

### CE-NF1-N05: Generalize needed reductions and numeric diagnostics
Reuse segment facilities for sum, norm, residual and weighted reduction needs with explicit empty/nonfinite behavior.

Acceptance: Empty and highly skewed segments, overflow controls and nonfinite cases match reference semantics.
Failure to reject: An unavailable reduction or numeric profile is not silently approximated or skipped.
Dependencies: CE-NF1-N04

### CE-NF1-N06: Publish general-width execution capability
Integrate actual host/V100 numeric paths and truthful capability inspection for storage, accumulation, widths, effects and derivative prerequisites.

Acceptance: Separate consumer tests exercise f32 and qualified f16 routes.
Failure to reject: Schema support without a linked provider cannot advertise an executable capability.
Dependencies: CE-NF1-N05

## H: Indexed n-ary mechanisms

### CE-NF1-H01: Define argument incidence and output ownership
Represent bounded indexed argument lists with roles, multiplicity, arity and output assembly; distinguish mechanism instance from a fabricated latent biological entity.

Acceptance: Repeated and ordered arguments retain semantics under physical reordering.
Failure to reject: Sorting incidence must not impose commutativity on an arbitrary evaluator.
Dependencies: CE-NF1-M10

### CE-NF1-H02: Implement grouped native evaluator registration
Register precompiled evaluators with forward, effects, dependencies, numeric policies and optional derivative actions; group dispatch by compatible evaluator/arity.

Acceptance: Consumer-defined evaluators use the same program and binding machinery.
Failure to reject: No per-cell Python or per-edge virtual callback enters the hot path.
Dependencies: CE-NF1-H01

### CE-NF1-H03: Implement independent CPU n-ary evaluation
Provide logical-order evaluation and destination assembly for the declared initial mathematical operations and arbitrary admitted interaction lists.

Acceptance: The nonlinear product of three inputs disagrees with an intentionally incorrect additive pairwise approximation.
Failure to reject: No dense Cartesian-product interaction enumeration is hidden in preparation.
Dependencies: CE-NF1-H02

### CE-NF1-H04: Execute n-ary groups on CUDA
Gather indexed arguments and evaluate/assemble in batched device code using shared structural indices and bounded scratch.

Acceptance: Irregular arity, empty activity and width tails agree with CPU reference.
Failure to reject: Materializing all cell-by-edge-by-argument values is forbidden unless explicitly selected and costed.
Dependencies: CE-NF1-H03, CE-NF1-N06, CE-NF1-P07

### CE-NF1-H05: Add exact inspectable compositions and custom block parity
Allow arithmetic composition and custom compiled blocks to represent the same known function with identical roles and output effects.

Acceptance: Both paths match values and preserve operation provenance across composition.
Failure to reject: A custom opaque block cannot claim dependency pruning or differentiation it has not supplied.
Dependencies: CE-NF1-H04

### CE-NF1-H06: Qualify deterministic assembly and nonfinite behavior
Test reduction order, conditional exclusion, repeated destinations and domain errors under the requested policy.

Acceptance: False predicates can avoid invalid branch evaluation when declared by the mathematical operation.
Failure to reject: Multiply-by-zero cannot silently substitute for conditional exclusion on NaN or invalid expressions.
Dependencies: CE-NF1-H05

### CE-NF1-H07: Publish n-ary execution receipt
Integrate the evaluator and group kernel with real Cellerator programs and independent tests; publish supported and unsupported capability set.

Acceptance: At least one truly combinatorial mechanism runs on V100 through the linked library.
Failure to reject: A pairwise-only facade or unlinked registration does not satisfy this capability.
Dependencies: CE-NF1-H06

## S: Support activity, compaction and structural epochs

### CE-NF1-S01: Integrate existing mask and indexed gate kernels
Connect byte/bit masks and indexed gates to the canonical program and generation contracts instead of creating GlassHelix-specific kernels.

Acceptance: Mask, values and topology have independent invalidation and visible cost.
Failure to reject: Missing gate dimensions and mask capacity are rejected before writing outputs.
Dependencies: CE-NF1-M10

### CE-NF1-S02: Distinguish shared and per-instance activity
Describe shared masks, cohort masks and cell-specific activity without presuming all states share the same active support.

Acceptance: Independent instances can evaluate different active sets over common structure.
Failure to reject: A global union mask cannot change per-instance semantics or leak candidate activity.
Dependencies: CE-NF1-S01, CE-NF1-N06

### CE-NF1-S03: Build a compact active-projection alternative
Add bounded selection/compaction with logical identity maps and a stable full-support fallback; charge selection, rebuild, mapping and execution costs.

Acceptance: Masked and compacted outputs agree on identical admitted operations.
Failure to reject: No speedup claim excludes activity recomputation or sparse projection construction.
Dependencies: CE-NF1-S02

### CE-NF1-S04: Separate primal and differential support
Retain trainable response routes even where a forward coefficient or contribution is numerically zero.

Acceptance: A zero weight in w*x has zero primal output and correct nonzero parameter derivative.
Failure to reject: Primal pruning cannot silently remove response support or probability mass.
Dependencies: CE-NF1-S03, CE-NF1-C03

### CE-NF1-S05: Implement safe topology replacement at boundaries
Reprepare only affected structures for genuine support edits; preserve old generations until outstanding users complete and publish new epochs explicitly.

Acceptance: Removing and restoring a relation produces expected updates and stale-ticket errors.
Failure to reject: Fast value-only updates must never rebuild topology; old maps cannot address the replacement.
Dependencies: CE-NF1-S04

### CE-NF1-S06: Add approximate dropping with accuracy provenance
Support explicit near-zero proposals tagged as bounded, empirical or unassessed, and refuse a requested bound when it cannot be supplied.

Acceptance: Pruned output error and eligibility domain are recorded separately from structural absence.
Failure to reject: A fitted near-zero value is not a causal absence claim or a universally valid reduction.
Dependencies: CE-NF1-S05

### CE-NF1-S07: Qualify churn and reuse regimes
Benchmark stable masks, persistent compact projections and rapidly changing masks with complete lifetime cost; retain losers as explicit candidates or reject them.

Acceptance: The selected path is explainable from measured benefit and requested semantics.
Failure to reject: Dynamic support is not declared universally faster; unsupported derivative paths fail visibly.
Dependencies: CE-NF1-S06

## D: Program-level directional differential execution

### CE-NF1-D01: Provide local arithmetic JVP and VJP
Implement derivative actions for the initial smooth numeric operations and register optional capabilities on the existing operation contracts.

Acceptance: Independent finite-difference and duality checks cover state and parameter directions.
Failure to reject: Forward-only blocks reject derivative requests instead of returning zero.
Dependencies: CE-NF1-M10

### CE-NF1-D02: Extend relation response through shared topology
Reuse forward/transpose and edge contraction work for local derivative actions with explicit value and operand generations.

Acceptance: Mixed/f32 policies are compared under their stated real-operation differentiation convention.
Failure to reject: Gradient-through-rounding is not silently inferred; f16 storage cannot erase an f32 requested perturbation.
Dependencies: CE-NF1-D01, CE-NF1-V07, CE-NF1-N06

### CE-NF1-D03: Implement n-ary evaluator derivatives
Evaluate supplied local JVP/VJP for ordered multi-input blocks, accumulating repeated-argument and shared-output contributions correctly.

Acceptance: Three-input product derivatives and repeated-role cases match independent analytic formulas.
Failure to reject: Pairwise surrogate derivatives or missing repeated-input factors fail tests.
Dependencies: CE-NF1-D02, CE-NF1-H07

### CE-NF1-D04: Compose forward directional programs
Build JVP stages for numerical dependency graphs and differentiate the actual composed map, preserving current branch and activity decisions.

Acceptance: Composed transition and observation JVP match finite differences for qualified smooth cases.
Failure to reject: A held-fixed mask derivative is not advertised as a derivative through a changing discrete selector.
Dependencies: CE-NF1-D03, CE-NF1-P07, CE-NF1-S07

### CE-NF1-D05: Compose reverse actions with explicit primal lifetime
Retain or checkpoint required primal values, reuse scratch legally and bind cotangents to the correct state/parameter snapshots.

Acceptance: VJP duality holds with fan-out, shared subexpressions and multiple outputs.
Failure to reject: Reusing a later parameter plane for an earlier rollout derivative is rejected.
Dependencies: CE-NF1-D04

### CE-NF1-D06: Support selected second-direction actions
Provide bounded second-order directional actions for the initial smooth vocabulary; make unsupported operations fail explicitly.

Acceptance: Analytic polynomial and nonlinear fixtures agree with independent second-direction references.
Failure to reject: No universal Hessian, automatic derivative of opaque C++ or infinite-jet claim is made.
Dependencies: CE-NF1-D05

### CE-NF1-D07: Validate nonsmooth boundaries and invalidation
Define selected branch-local behavior only where mathematically valid and track unsupported points, support epochs and failed primal executions.

Acceptance: Boundary and stale-primal tests produce explicit diagnostic status.
Failure to reject: Numerical finite differences remain labeled approximation, not a silent substitute for an exact action.
Dependencies: CE-NF1-D06

### CE-NF1-D08: Publish differential execution receipt
Qualify linked host and GPU derivative paths, widths, counters, memory use and negative cases for downstream GlassHelix import.

Acceptance: Receipt binds exact differential contracts and independent executable evidence.
Failure to reject: Metadata validation, CPU-only surrogate or declaration syntax checks cannot satisfy GPU differentiation.
Dependencies: CE-NF1-D07

## T: Independent validation and regression qualification

### CE-NF1-T01: Author independent formula and indexing referees
Build small f64 formulas and separately constructed logical maps for numeric, n-ary and derivative operations, independent from production lowering.

Acceptance: Deliberate endpoint permutation and omitted derivative terms are detected.
Failure to reject: Matching two paths that share the same faulty map is not independent validation.
Dependencies: CE-NF1-M00

### CE-NF1-T02: Author shape, numerical and nonfinite stress cases
Cover empty support, high-degree segments, argument multiplicity, tails, exact masks, near-zero response and requested precision policies.

Acceptance: Tests state exact versus tolerance-based expected behavior explicitly.
Failure to reject: Nonfinite inputs cannot be silently sanitized into a passing biological interpretation.
Dependencies: CE-NF1-T01

### CE-NF1-T03: Author asynchronous lifetime and failure tests
Exercise ownership, overwritten borrowed arrays, late invalid binding, stale epochs, reader completion and explicit poisoned execution.

Acceptance: Unsafe mutations and stale tickets produce deterministic failure before valid-result publication.
Failure to reject: Launch acceptance alone cannot satisfy completion or sanitizer acceptance.
Dependencies: CE-NF1-T02

### CE-NF1-T04: Qualify external custom operations and capability honesty
Compile a separate custom operation consumer, including one forward-only block, and audit reported capabilities against actual providers.

Acceptance: Host and device execution work where advertised and unsupported requests fail closed.
Failure to reject: Header presence, CMake target existence and fake CPU fallback do not count as GPU support.
Dependencies: CE-NF1-T03, CE-NF1-M30

### CE-NF1-T05: Run retained and new integrated conformance
Execute relevant SS1/RU1, prepared-program, gate/segment and consumer regressions affected by source changes using current build outputs.

Acceptance: All required cases execute, with skips and missing binaries treated as failures.
Failure to reject: No old test is deleted or weakened merely to make a changed contract pass.
Dependencies: CE-NF1-T04, CE-NF1-M40

### CE-NF1-T06: Run sanitizer and resource qualification
Use the existing native GPU lease plus the shared NF1 performance lock; run compute sanitizer and separate correctness/timing campaigns.

Acceptance: Successful receipts record exact source, binary, tool, device and zero-error summary.
Failure to reject: Missing hardware, contaminated timing and expected-failure process modes cannot become passing accelerator evidence.
Dependencies: CE-NF1-T05

## O: Whole-program optimization and measured promotion

### CE-NF1-O01: Establish composed baseline and complete cost records
Benchmark complete prepared programs with setup, transfers, value refresh, support changes, forward/response stages and required result observation.

Acceptance: Benchmark receipts distinguish cold, resident and amortized lifetime cost.
Failure to reject: Kernel-only medians or synthetic counts cannot be presented as application speedup.
Dependencies: CE-NF1-M30

### CE-NF1-O02: Hoist reuse and preserve prepared orders
Reuse structural maps and legal packs by full generation/dependency keys; remove only unnecessary canonicalization, not evidence identity.

Acceptance: Repeated states and mechanisms reduce measured packing/movement without changing outputs.
Failure to reject: Same address or same evaluator name cannot justify cache reuse after a value update.
Dependencies: CE-NF1-O01, CE-NF1-M40

### CE-NF1-O03: Fuse indexed nonlinear evaluation and assembly
Prototype grouped gather/evaluate/reduce fusion under explicit output, nonfinite and derivative semantics; retain unfused reference.

Acceptance: Real program latency or moved bytes improve in a measured regime.
Failure to reject: Register pressure, lost ordering or changed predicates disqualify a purported optimization.
Dependencies: CE-NF1-O02

### CE-NF1-O04: Specialize legal batch and relation routes
Compare existing sparse, dense-fragment, vendor and width-specialized routes under matched numerical policy and topology reuse.

Acceptance: Candidate choice reflects full cost and supports tail widths.
Failure to reject: No universal WMMA promotion or unsupported newer GPU feature is assumed.
Dependencies: CE-NF1-O03

### CE-NF1-O05: Measure active support crossover and precision tradeoffs
Sweep active fraction, churn and reuse; separately assess f32, qualified mixed precision and approximate dropping.

Acceptance: Report crossover evidence and error assessment with a retained fallback.
Failure to reject: A faster approximate computation is not mislabeled an exact implementation speedup.
Dependencies: CE-NF1-O04

### CE-NF1-O06: Qualify bounded read-only replay where supported
Add or qualify repeated fixed-program replay only for capture-safe operations after ordinary stream execution is correct.

Acceptance: Supported replay produces current input-bound results and measurable evidence; unsupported mutable capture remains explicit.
Failure to reject: No host generation counter is falsely advanced by graph replay; capture is not required for unrelated correctness.
Dependencies: CE-NF1-O05

### CE-NF1-O07: Publish non-promotion or promotion decision
Summarize baseline comparisons, uncertainty, device/build identity, peak memory and admitted regimes; publish capability-ready receipt even when optional optimizations do not win.

Acceptance: The planner consumes justified choices and preserves correct alternatives.
Failure to reject: A negative performance result is retained; workload arithmetic cannot be changed to manufacture a win.
Dependencies: CE-NF1-O06

## X: Cross-repository acceptance import

### CE-NF1-X01: Verify GlassHelix real-consumer receipt
Import the GH-NF1-M60 receipt only after checking producer UUID/run/task, integrated source commit, contract hashes, Cellerator dependency revision and real test receipts.

Acceptance: The consumer used the exact Cellerator capability bundle being closed or an explicitly qualified descendant.
Failure to reject: A pending template, stale build, mismatched interface or a reference-only demo does not unlock final Cellerator closure.
Dependencies: CE-NF1-M50

## M: Central integration and release

### CE-NF1-M00: Baseline and ownership accepted
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Accept source disposition, tool binding, resource policy and preserved-work boundaries.

Acceptance: Accept source disposition, tool binding, resource policy and preserved-work boundaries.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-A04

### CE-NF1-M10: Developmental contracts published
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Publish CE-NF1 capability contracts with versioned source hashes for GlassHelix X01.

Acceptance: Publish CE-NF1 capability contracts with versioned source hashes for GlassHelix X01.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M00, CE-NF1-C06

### CE-NF1-M20: Linked execution and numeric core published
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Host and V100 consumer run shared-structure multi-operand programs with f32 and general width.

Acceptance: Host and V100 consumer run shared-structure multi-operand programs with f32 and general width.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M10, CE-NF1-B06, CE-NF1-P07, CE-NF1-V07, CE-NF1-N06

### CE-NF1-M30: Combinatorial and support execution published
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Publish true n-ary and active-support capability, full fallback and structural-epoch conformance.

Acceptance: Publish true n-ary and active-support capability, full fallback and structural-epoch conformance.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M20, CE-NF1-H07, CE-NF1-S07

### CE-NF1-M40: Directional differential execution published
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Publish linked state/parameter JVP/VJP, selected second directions and stale-primal rejection.

Acceptance: Publish linked state/parameter JVP/VJP, selected second directions and stale-primal rejection.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M30, CE-NF1-D08

### CE-NF1-M50: Consumer-ready qualified capability bundle
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Publish qualified capability, regression, sanitizer and complete-cost evidence for GH final integration.

Acceptance: Publish qualified capability, regression, sanitizer and complete-cost evidence for GH final integration.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M40, CE-NF1-T06, CE-NF1-O07

### CE-NF1-M90: Cellerator foundation closed and integrated
Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Integrate all NF1 work, preserve retained consumers and publish final receipt before GH final closure.

Acceptance: Integrate all NF1 work, preserve retained consumers and publish final receipt before GH final closure.
Failure to reject: Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.
Dependencies: CE-NF1-M50, CE-NF1-X01
