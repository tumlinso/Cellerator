# Cellerator: research findings and recommended direction

**Cellerator has a credible opportunity to accelerate biological mathematics by making structure useful throughout evaluation, differentiation, updating, and interrogation. The strongest direction is a native library of reusable structured operations, built on established numerical libraries, with specialized representations and algorithms where they demonstrably remove work.**

The evidence supports some bounded computational savings. It does **not yet establish that those savings survive complete biological-model construction, learning, and interrogation**. The report makes that distinction explicit and lays out experiments capable of resolving it.

The complete report is **34 pages**, with mathematical reasoning, implementation comparisons, an experimental program, and a source register:

- [Cellerator_Research_Report.pdf](https://chatgpt.com/library?file_id=libfile_fde3_f-a547ff7f72b08191a5bc0178a2e7b241_532ddaf90af88191920701d35b04bae9&account_id=personal) — complete report.
- [Cellerator_Research_Report.docx](https://chatgpt.com/library?file_id=libfile_fde3_f-d8209d8261948191a61fce15cf2248aa_532ddaf90af88191920701d35b04bae9&account_id=personal) — editable version.
- [Cellerator_Evidence.zip](https://chatgpt.com/library?file_id=libfile_fde3_f-121df1d6a29881919233550b7b971dd8_532ddaf90af88191920701d35b04bae9&account_id=personal) — inspected source snapshots, hashes, retrieval boundaries, and reference register.

Below are the substantive conclusions and the reasoning behind them. **“Finding” identifies inspected evidence; “derivation” identifies a consequence of stated assumptions; “recommendation” identifies my judgment about what to do next.**

## 1. What the existing Cellerator work actually establishes

### The cellular geometry ideas are a useful foundation

I could access Cellerator and read the cellular geometry document. It already addresses several of the most consequential distinctions: admissible dependencies versus observed activity; context-dependent eligibility; logical identity through packing; instantaneous versus finite-time influence; and the fact that a sparse Jacobian does not uniquely determine the underlying mechanism decomposition. Those distinctions are directly relevant to safe computational omission. cellular_geometry.md[Inspected cellular geometry document](sandbox:/workspace/scratch/09d0ae2d4715/research/cellerator_snapshot/cellular_geometry.md)

**My interpretation:** Cellerator’s numerical responsibility is to express **what a declared model permits the machine to omit, under which conditions, and for which requested calculation**. Biological knowledge, learned structure, numerical approximation, and a temporary inactive branch can all influence execution, but they justify different omissions.

The native data documentation already separates identity, dependency geometry, storage ownership, encodings, execution projections, operations, and retained numerical generations. It also describes borrowed ATen integration and bounded distributed execution. Much of the right conceptual boundary therefore exists already. native_data.qmd[Native data and execution contracts](sandbox:/workspace/scratch/09d0ae2d4715/research/cellerator_snapshot/docs/native_data.qmd)

### There is encouraging performance evidence, with narrow boundaries

The most informative inspected result is the shared-product prototype:

- Resident forward plus all five requested gradient outputs fell from **50.8444 to 47.7281 microseconds**: **6.13% less time**.
- The comparison uses a closely matched unfactored control.
- Both experimental arms already reuse the common FP32 product.
- The measurement excludes construction, uploads, allocation, downloads, and optimizer updates.
- Its seven paired blocks come from one execution; they do not establish independent-run or workload variability.

This is useful evidence that an algebraic transformation can help the combined forward/backward computation. It is not an end-to-end model result. The much larger difference against the general native path also involves a substantial memory/specialization tradeoff and cannot be attributed solely to factorization. REPORT.md[Prototype report and measurement scope](sandbox:/workspace/scratch/09d0ae2d4715/research/cellerator_snapshot/bench/factorized_interactions/prototypes/shared_product_circuit/REPORT.md)

An older shared-support experiment reports complete iteration improvements against a materialized Torch implementation, including loss, backward, Adam, and coefficient publication. Its missing decisive competitor is a **factorized library implementation using sparse multiplication, transpose multiplication, and sampled parameter-gradient computation**. Its native peak-memory accounting is also incomplete. README.md[Shared-support evidence](sandbox:/workspace/scratch/09d0ae2d4715/research/cellerator_snapshot/docs/learning/bio-evidence/README.md)

**Recommendation:** Preserve these results as evidence for specific hypotheses. Build the stronger competitors and complete the lifecycle accounting before promoting a general speedup claim.

## 2. What to borrow, and what Cellerator should own

### Borrow mature numerical machinery

| Need | Strong starting implementation | Cellerator’s contribution |
|---|---|---|
| Ordinary tensors and framework integration | LibTorch/ATen | Borrowed views, mutation rules, ownership, streams, and custom operation contracts |
| Dense and conventional sparse arithmetic | cuBLAS/cuBLASLt, cuSPARSE | Prepared structure, valid dispatch, parameter identity, and coupled forward/backward execution |
| Packing and work construction | Compatible CUB/CCCL components | Work keys, masks, inverse maps, supplied capacity, and invalidation |
| Specialized dense tiles | Selected CUTLASS providers | Traversal, fusion, or numerical specialization missing from a suitable library call |
| Numerical codecs | ggml, FBGEMM, selected bitsandbytes components | Sparse topology, feature semantics, overrides, updates, and differentiation policy |
| Sparse and implicit solves | PETSc, Ginkgo | Residuals, derivative actions, and useful structural preconditioner information |
| Mechanistic dynamics and sensitivities | AMICI/SUNDIALS | Efficient local laws, batching, and native adapters |
| Pairwise reductions and distributional losses | KeOps, GeomLoss, appropriate transport implementations | The native integration boundary and the scientific query being computed |

These are concrete implementation families, with different contracts and compatibility requirements. cuSPARSE supplies conventional sparse primitives; CUTLASS supplies configurable dense kernels; AMICI already combines native biological simulation with sensitivity analysis; PETSc and Ginkgo supply extensive solver machinery. None requires Cellerator to rebuild a general tensor or solver stack. [cuSPARSE 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html?utm_source=chatgpt.com)

Some valuable borrowing is algorithmic rather than a direct dependency. GraphBLAS provides structural-mask and algebraic-operation precedents; taco and Finch explore structure-aware code generation; factorized database work demonstrates avoiding expanded combinations and maintaining affected computations. CppAD supplies sparse differentiation and coloring. These deserve consideration without requiring a wholesale move into another runtime. [GitHub](https://github.com/DrTimothyAldenDavis/GraphBLAS?utm_source=chatgpt.com)

### Own the information ordinary numerical libraries cannot infer

Cellerator should own:

- Logical entities, mechanism roles, repeated argument occurrences, and parameter sharing.
- Which dependencies are forbidden, merely unobserved, provisionally excluded, or currently inactive.
- Which inputs and derivatives a particular query requires.
- How canonical values map into forward and reverse execution layouts.
- Which structural changes invalidate preparation.
- Which actual forward values must remain alive for backward.
- What an approximation claims to preserve, and when a stricter route is required.

These can remain concrete native descriptors, supplied buffers, and deliberate operation calls. They do not imply a universal model hierarchy or another general optimizer.

A central distinction is between **shared support**, **shared formula**, and **shared evaluated value**. Two cells can share a sparse pattern and a mechanism law while having different operands and gradients. Exact numerical reuse requires matching the relevant operands, parameters, context, and generation. Approximate similarity introduces a separate error contract.

## 3. The strongest opportunities begin with the calculation

### A. Factor shared relations before expanding cell-edge values

For the existing shared-support law, write a cell’s computation as

\[
q_b=s_b\odot x_b,\qquad
v_b=Wq_b,\qquad
y_b=a_b\odot v_b.
\]

For output cotangent \(\lambda_b\),

\[
u_b=a_b\odot\lambda_b,\qquad
t_b=W^\top u_b,
\]

and an edge parameter receives

\[
\frac{\partial L}{\partial w_e}
=\sum_b u_{b,\operatorname{dst}(e)}
       q_{b,\operatorname{src}(e)}.
\]

**Derivation:** The required work is a forward sparse-dense multiplication, a transpose multiplication, a sampled product on the edge support, and elementwise operations. A separate value or message for every cell-edge pair is unnecessary for this law.

This gives a strong conventional baseline and a possible fusion target. PyTorch Geometric documents the related opportunity to eliminate explicit messages when a formula admits sparse matrix aggregation. [cuSPARSE 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html?utm_source=chatgpt.com)

The model restriction matters: two-sided diagonal modulation does not express every changing relation or arbitrary edge-conditioned attention. Compare implementations while holding the equations fixed; compare expressivity separately.

### B. Share mechanism calculations and their reverse reductions

Products with repeated or zero-valued arguments need correct derivatives. Cellerator already uses zero-safe prefix/suffix products. The next question is when shared factors should be computed once, retained, and differentiated after accumulating all incoming adjoints.

**Recommendation:** Extend the existing matched experiment across reuse count, arity, output overlap, parameter tying, and save-versus-recompute choices. A shared intermediate can save arithmetic while creating memory traffic or a reduction hotspot. Those competing effects determine the useful regime.

For reaction networks,

\[
f(x)=N\,v(x)
\]

also separates a sparse incidence/stoichiometric operation from nonlinear local laws. When \(N\) is fixed for the derivative being taken, forward sensitivities apply \(N\) to local rate sensitivities; reverse sensitivities begin with \(N^\top\). If \(N\) is learned or state-dependent, its derivative contributes additional work. AMICI is an appropriate established biological reference for this family. [AMICI 1.2.0 documentation](https://amici.readthedocs.io/en/latest/about.html?utm_source=chatgpt.com)

### C. Compute derivative actions before materializing Jacobians

For residual Jacobian \(J\), an iterative method can apply

\[
v\mapsto J^\top(Jv)+\lambda v
\]

using a JVP and VJP without storing \(J\) or its normal matrix. Both actions must use the same linearization point, parameter generation, topology, and relevant random realization.

When sparse entries really are needed, established coloring methods can compress the required derivative directions. CppAD already provides forward/reverse sparse-Jacobian machinery and reusable work objects. [CppAD documentation](https://cppad.readthedocs.io/latest/sparse_jac.html?utm_source=chatgpt.com)

Cellerator should explicitly publish which operations support primal evaluation, VJP, JVP, batching, and higher derivatives. A custom backward alone does not establish all these capabilities; PyTorch treats them as distinct extension obligations. [PyTorch 2.15 documentation](https://docs.pytorch.org/docs/2.15/notes/extending.func.html?utm_source=chatgpt.com)

### D. Reuse solvers, and optimize what they request

Equilibrium modules, implicit layers, fate probabilities, and mechanistic fitting often need linear solves rather than an explicitly formed inverse or full derivative tensor.

**Recommendation:** Expose residual and derivative-action interfaces, then connect appropriate established solvers. Investigate preconditioners explicitly: reducing storage while greatly increasing iteration count can lose.

Checkpointing deserves the same complete accounting. Replaying an operation can repeat gathers, decoding, packing, transfers, and synchronization—not only arithmetic. Continuous sensitivities, derivatives of a discrete integrator, and event/reset sensitivities also have different contracts. SUNDIALS already supplies checkpointed adjoint machinery, while hybrid dynamical events may require event-time corrections such as saltation matrices. [User Documentation for SUNDIALS documentation](https://sundials.readthedocs.io/en/v7.3.0/cvodes/Usage/ADJ.html?utm_source=chatgpt.com)

## 4. ggml and quantization: useful components, distinct training contracts

### Cellerator already has a ggml starting point

The inspected integration pins ggml, disables its GPU backends, and provides CPU rowwise numerical encode/decode. Its documented qualification is a focused harness. That is a useful existing boundary to extend deliberately. ggml.qmd[Existing ggml integration](sandbox:/workspace/scratch/09d0ae2d4715/research/cellerator_snapshot/docs/ggml.qmd)

Upstream ggml provides defined block layouts, reference codecs, row-size rules, and packed consumers. Its block metadata changes the actual storage cost:

| Format | Values per block | Bytes per block | Effective bits/value |
|---|---:|---:|---:|
| Q4_0 | 32 | 18 | 4.5 |
| Q4_1 | 32 | 20 | 5.0 |
| Q8_0 | 32 | 34 | 8.5 |
| Q2_K | 256 | 84 | 2.625 |
| Q4_K | 256 | 144 | 4.5 |

These figures follow the upstream structs and exclude outer alignment and training state. Borrow the complete pinned codec contract, including scale metadata and grouping constraints. [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h?utm_source=chatgpt.com)

For a relevant operation, compare three routes:

1. Consume packed values directly.
2. Decode once and reuse an ordinary-precision representation.
3. Keep an ordinary-precision representation throughout.

The upstream build guide itself identifies a V100 tradeoff where custom quantized multiplication can save memory while FP16 cuBLAS is faster at larger batches. Packed execution is therefore a candidate, not an automatic winner. [raw.githubusercontent.com](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md?utm_source=chatgpt.com)

### Other implementations worth using

**FBGEMM** offers established affine rowwise codecs and embedding-operation precedents. Its fused update machinery is relevant where that update ordering matches the intended training semantics. [FBGEMM 1.5.0 documentation](https://docs.pytorch.org/FBGEMM/fbgemm/cpp-api/QuantUtils.html?utm_source=chatgpt.com)

**bitsandbytes** provides useful quantization and optimizer-state baselines. Its inspected current 4-bit forward uses fused quantized multiplication, while backward decodes weights for the input gradient and does not supply an ordinary packed-weight gradient. **QLoRA** addresses a frozen quantized base with trainable adapters; that is narrower than arbitrary trainable biological coefficients and states. [GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py?utm_source=chatgpt.com)

**GPTQ, AWQ, and HQQ** contribute approximation/calibration methods; **Marlin** contributes a specialized inference kernel. They should be evaluated for the particular workload rather than treated as interchangeable storage standards. **Transformer Engine** supplies more directly relevant low-precision training recipes on supported modern hardware. [GitHub](https://github.com/IST-DASLab/gptq?utm_source=chatgpt.com)

**zfp and SZ3** belong in the investigation of trajectories, checkpoints, and communication buffers. Their numerical reconstruction guarantees do not automatically guarantee accurate gradients, event times, or rare-state probabilities. [zfp 1.0.1 documentation](https://zfp.readthedocs.io/en/latest/modes.html?utm_source=chatgpt.com)

### Count the representation and training state together

Two illustrative derivations show why value compression can disappoint:

- CSR with a four-byte index and four-byte value uses approximately **eight bytes per nonzero** before row pointers. Replacing the value with four bits gives **at least 4.5 bytes**, only about **1.78×** smaller before scales and other metadata.
- An FP32 parameter, gradient, and two Adam moments use **sixteen bytes per parameter**. Compressing only the parameter to four bits leaves **12.5 bytes**, about **1.28×** smaller before metadata. Retaining a master parameter changes that accounting again.

For Cellerator’s state codes, distinguish fixed codes with trainable codebook entries from learning the discrete code selection. The first has ordinary gather/reduction derivatives; the second needs an explicit discrete or surrogate policy. Exact overrides also need defined precedence and update behavior.

**Recommendation:** Prioritize quantization where it removes a measured bottleneck in the complete consumer. Optimizer state, saved state, and communication may offer greater total benefit than compressing an already small sparse value array.

### Use descriptive names

The report proposes names such as **row block bitmap records**, **row-tiled block bitmap**, **feature codebook encoding**, **CSR with packed affine values**, and **dense-tile/CSR-residual projection**.

The purpose is to expose topology, encoding, and traversal separately. A warp-sized implementation choice should remain a provider detail where possible. These are naming directions, not a proposed rename patch or finalized ABI.

## 5. How biological knowledge should remove work during training

The useful question is **which cost a piece of knowledge actually eliminates**.

| Knowledge or stage | Work it can remove | Work that may remain |
|---|---|---|
| Declared forbidden dependency | Candidate construction, absent parameters, corresponding derivatives | Checking that the restriction applies to the modeled regime |
| Fixed experimentally supported coefficient | Its parameter gradient and optimizer state | Input/context derivatives through the module |
| Known topology, uncertain kinetics | Repeated structural discovery and indexing | Coefficient and activity learning |
| Early provisional learned support | Some local evaluation | Exploration, growth probes, and reactivation |
| Stable learned structure | Repeated sorting, reverse-map construction, coloring | Numeric refresh and validation |
| Selected interrogation | Unrequested gradients and optimizer work | Every path required by the scientific query |

**A fixed module can still transmit gradients.** Freezing its parameters and detaching its inputs are different operations. The latter changes what upstream components can learn.

Similarly, differentiating the current sparse graph cannot discover every omitted edge. Support learning needs candidate generation and a route for growth. RigL is one existing computational policy worth comparing, without interpreting its learned sparsity as proof of biological absence. [proceedings.mlr.press](https://proceedings.mlr.press/v119/evci20a.html?utm_source=chatgpt.com)

Optimizer semantics also matter. With momentum, Adam, or weight decay, a zero gradient can still lead to state or parameter changes. PyTorch explicitly distinguishes a zero gradient from an absent gradient. Skipping updates must preserve the intended optimizer or be declared as a changed training policy. [PyTorch 2.15 documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html?utm_source=chatgpt.com)

Your concern about batch effects changes the validation requirements. A regular technical signature can be computationally convenient yet be a poor reason to exclude a biological dependency permanently. Lossless scheduling may still exploit such regularity without changing the model; the problematic step is treating it as biological evidence.

**Recommendation:** Preserve the source and context of structural constraints. Validate learned exclusions across appropriate donors, studies, conditions, perturbations, and lineages. If condition and batch are perfectly confounded, the observations alone cannot identify their separate contributions without additional information or assumptions.

Statistical equivalence must also match the intended use. Similar average predictions do not establish similar sensitivities, intervention rankings, rare-state probabilities, or dynamical stability. Predeclare equivalence margins; failure to detect a significant difference is insufficient.

## 6. Scientific consumers should expose reusable numerical demands

I would use a diverse suite, while keeping scientific preprocessing and interpretation in the consuming projects.

| Scientific consumer | Numerical capability it tests |
|---|---|
| Cellular dynamics and transitions | Local rate laws, integration, JVP/VJP, sparse solves |
| Unseen and combined perturbations | Conditional mechanisms, shared factors, sensitivity queries |
| Lineage and history | Ragged forests, shared prefixes, reverse tree reductions |
| Population transport and spatial coupling | Pairwise reductions, log-sum-exp, Sinkhorn, neighborhood operators |
| Multimodal latent state | Observation likelihoods, missing-modality masks, conditioned dense blocks |
| Interrogation and experimental design | Selected derivatives, curvature actions, implicit solves, ensemble reductions |

Three examples are especially revealing.

**First, the question itself can remove work.** CellRank 2 aggregates terminal representatives before solving for requested fate populations, instead of solving more individual problems and aggregating afterward. That is an established exact query-level optimization and a strong precedent for Cellerator. [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239496/?utm_source=chatgpt.com)

**Second, sparse observations need not imply sparse loss computation.** For a negative-binomial observation with mean \(\mu>0\) and inverse dispersion \(\theta>0\),

\[
-\log p(0\mid\mu,\theta)=\theta\log(1+\mu/\theta),
\]

whose derivative with respect to \(\mu\) is positive. Dropping measured zeros changes the objective. A useful acceleration could instead fuse decoder, likelihood, and backward reductions while avoiding large retained intermediates. scVI supplies a relevant reference consumer. [scvi-tools](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/scvi.html?utm_source=chatgpt.com)

**Third, histories can expose shared computation.** Exact ancestral prefixes can be evaluated once and receive accumulated child adjoints when modeled state, parameters, context, and stochastic realization agree. Reconstructed ancestry alone does not establish identical ancestral states. Existing lineage resources provide concrete workloads, while BEAGLE supplies a mature native precedent for compatible tree-likelihood computations. [GitHub](https://github.com/beagle-dev/beagle-lib?utm_source=chatgpt.com)

The foundation-model evidence also argues for disciplined comparisons. A 2025 perturbation study found strong simple linear baselines; a 2026 study found that calibrated metrics exposed improvements missed by common measures; another 2026 study found performance saturation in several tasks across a large pretraining experiment. These are bounded findings about particular evaluations, not a universal verdict for or against foundation models. [doi.org](https://doi.org/10.1038/S41592-025-02772-6?utm_source=chatgpt.com)

Promising exploratory directions include uncertainty-aware ensembles sharing compatible computation, rare-transition interrogation, sensitivity-guided precision, and selecting perturbations through repeated derivative actions. Their scientific novelty remains unestablished; their reusable mathematical demands can be investigated directly.

## 7. V100 and distributed execution need explicit compatibility and cost models

**One immediate finding:** CUDA 13 removed offline compilation and library support for Volta. Preserve a reproducible CUDA 12.x V100 configuration and qualify the exact LibTorch, compiler, and provider combination. A current package name does not imply SM70 support. [Release Notes 13.0 documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html?utm_source=chatgpt.com)

Newer providers have different boundaries: current Triton documents CC80+, modern FBGEMM default packages omit SM70, and cuSPARSELt targets newer hardware. Selected CUTLASS components remain relevant to SM70 with a compatible toolchain. [GitHub](https://github.com/triton-lang/triton?utm_source=chatgpt.com)

On V100, memory access, metadata traffic, reductions, and launch overhead are plausible priorities. Nevertheless, dense Tensor Core work can outperform a smaller amount of irregular sparse arithmetic. Measure the crossover rather than deriving it from sparsity alone.

For distributed execution, reuse NCCL transport machinery and choose partitions using **forward values, reverse cotangents, tied-parameter reductions, latency, and load imbalance**. Edge count alone misses reuse and backward synchronization hotspots. [NCCL 2.32.3 documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html?utm_source=chatgpt.com)

The existing two-device operation offers a concrete experiment: mixed activity currently transfers the full prepared projection with zero placeholders. Active compaction could help, but its scans, count exchange, packing, inverse maps, and reverse scattering must all be charged.

Two-device success would be first-stage evidence. General distributed claims require strong and weak scaling across the actual NVLink topology and any intended inter-node configuration.

## 8. The experiments that can resolve the principal question

**The central comparison should cover construction through training and interrogation.**

For a specialization,

\[
\text{net time saved}
=
\sum_{\text{uses}}
(\text{reference complete cost}-\text{specialized complete cost})
-\text{extra initial preparation}.
\]

“Complete cost” includes eligibility discovery, grouping, packing, derivatives, reductions, optimizer work, publication, transfers, synchronization, rebuilds, and audits. Preparation is repaid only if sufficient useful reuse follows.

Peak memory must likewise include topology, address maps, reverse projections, saved values, gradients, optimizer moments, communication staging, and temporary workspace—counting only allocations simultaneously live.

### Recommended implementation sequence

| Stage | Concrete result |
|---|---|
| **1. Establish the baseline** | Strong factorized relation baseline, matched product circuits, complete memory ledger, reproducible V100 build |
| **2. Complete a reusable training slice** | Forward, demanded derivatives, updates, retained lifetimes, checkpoint restore, and one scientific query |
| **3. Make conditional work economical** | Measured predicate/worklist/rebuild costs, support reactivation, justified communication compaction |
| **4. Connect interrogation and solvers** | Fixed-generation derivative actions and one established solver consumer, including a difficult counter-case |
| **5. Evaluate approximation and modern GPUs** | Quantization, compression, and precision policies measured against scientific quality targets and hardware-appropriate competitors |

For each stage, distinguish:

- **Implementation comparisons:** hold equations, parameter identities, optimizer semantics, and numerical acceptance fixed.
- **Approximation comparisons:** allow changed precision, support, tolerances, or model family, then measure time to quality and scientific error.

Derivative checks should include analytic or high-precision references, directional checks, zeros, repeated arguments, parameter ties, empty work, and changing support. An adjoint identity alone is insufficient because mutually consistent wrong implementations can satisfy it.

Performance evidence should include repeated independent sessions, paired comparisons, actual route selection, native and framework allocations, retained failures, and source/binary provenance.

Scientific tests should hold out whole donors, studies, perturbations, combinations, clones, or embryos when those are the claimed generalization targets. Millions of cells from one donor do not provide millions of independent biological replicates.

**A candidate should stop growing when its strongest valid baseline wins across its intended regime or its approximation fails the declared scientific target.** A useful narrow implementation—or a reliable native integration of an existing algorithm—is still a successful outcome.

## 9. Where no universal answer is justified

There is presently no defensible universal choice of sparse format, bit width, hardening point, model architecture, or device partition.

That uncertainty is actionable:

- **Format choice:** measure a selection surface over shape, skew, width, reverse access, and structural churn.
- **Precision choice:** test the actual observables and derivatives, including sensitive and rare regimes.
- **Hardening:** compare avoided work with discovery, validation, and maintenance costs; retain reactivation where the model permits it.
- **Architecture:** compare scientific capability and time to quality without silently restricting expressivity.
- **Distributed placement:** account for both directions and shared-parameter reductions.
- **Biological identification:** retain uncertainty where observations and prior knowledge cannot distinguish mechanisms.

My strongest recommendation is to make **a complete, reusable training-and-interrogation operation** the next unit of progress. Use block-masked storage to test the representation boundary, shared relations and product mechanisms to test learning, and selected sensitivities or fate solves to test interrogation. This connects the scientist’s declared structure to measurable avoided work while leaving room for substantially different algorithms.

## Integrity audit

**Confirmed limitations:** This investigation includes source inspection, primary-source research, mathematical derivations, and a concrete experimental roadmap. It contains no new GPU benchmark, qualified dependency build, biological training run, or demonstrated end-to-end Pareto improvement. The inspected Cellerator worktree was dirty; the evidence archive therefore preserves per-file provenance.

**Plausible remaining gaps:** The source review was targeted, not an exhaustive call-graph audit. Stochastic differentiation, distributed preconditioning, spatial mechanics/PDEs, non-backpropagation learning, and non-NVIDIA execution received less depth than structured native training. Actual consumer traces and hardware topology are still needed to select winning implementations.

**Potential narrowing:** Existing layouts could bias the agenda toward current code, while the large inference-quantization literature could overemphasize frozen weights. The report counters those tendencies with algebraic alternatives, solver reuse, backward costs, optimizer semantics, strong baselines, and explicit stopping criteria. I found no evidence supporting a claim of hidden steering. These checks have detection limits and cannot certify completeness.
