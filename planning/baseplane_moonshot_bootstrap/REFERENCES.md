# Research sources

Primary sources consulted on 2 October 2026. Source statements below are deliberately narrow. All Baseplane transfers, new mechanisms and combinations in this package are design proposals, not results established by these papers. This is a targeted research synthesis, not a systematic novelty survey.

## S01 — NVIDIA Volta Tuning Guide, CUDA 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html

Official architecture constraints: independent thread scheduling, resource hierarchy and Tensor Cores. Use for legality and occupancy questions, not a prediction that a particular kernel is fast.

Inspected: Sections 1.4.1–1.4.3.

## S02 — NVIDIA PTX ISA 8.8

https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html

Target notes establish sm_70 match.sync, eligible packed integer and Boolean operations, and FP16 mma.m8n8k4. Later-generation operations must not leak into the Volta baseline.

Inspected: Sections 9.7.1.23, 9.7.8.6, 9.7.13.10, 9.7.14.5 and target ISA notes.

## S03 — NVIDIA CUDA C++ Programming Guide 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html

Defines synchronized warp participation, opaque WMMA fragments, alignment and stride requirements, and finite-precision texture filtering. These are implementation contracts, not performance claims.

Inspected: Warp matrix functions; Texture and Surface Memory; Linear Filtering; Compute Capability 7.x.

## S04 — Jia et al. (2018), Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking

https://arxiv.org/abs/1804.06826

Empirical disassembly and microbenchmarks expose native Volta execution details beyond high-level APIs. Their observed mappings motivate inspection; they are not a portable fragment ABI or a promise about our binaries.

Inspected: PDF inspected, including tensor-core section p.41 / PDF page 42.

## S05 — NVIDIA CUDA Binary Utilities 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-binary-utilities/index.html

Official cuobjdump/nvdisasm reference for inspecting emitted native instructions. Compilation/disassembly can answer instruction questions without launching a GPU benchmark.

Inspected: cuobjdump and nvdisasm.

## S06 — NVIDIA (2025), Navigating GPU Architecture Support

https://developer.nvidia.com/blog/?p=102958

CUDA 13 removes offline compilation for architectures before compute capability 7.5. Retain an sm_70-capable toolkit; this package targets the repository’s CUDA 12.9 route.

Inspected: GPU support and developer guidance.

## S07 — Gu and Dao (2023), Mamba

https://arxiv.org/abs/2312.00752

Input-dependent state-space parameters and hardware-aware scan provide a useful precedent for sequence-dependent state evolution. Transfer the effect-composition question, not an assumption that every neural update admits a small associative summary.

Inspected: Full HTML v2 examined.

## S08 — Hwang, Wang and Gu (2025), H-Net

https://arxiv.org/abs/2507.07955

Dynamic chunking jointly learns hierarchical segmentation with sequence modeling. This supports treating boundaries as learned computation rather than a supplied genome annotation.

Inspected: Full HTML v1, architecture and routing sections.

## S09 — Pagnoni et al. (2024), Byte Latent Transformer

https://arxiv.org/abs/2412.09871

Entropy-based variable patching reallocates computation over a byte stream. It suggests a cheap initial routing baseline; predictive surprise is not itself biological importance.

Inspected: Abstract and primary publication record.

## S10 — Sweldens (1995), The Lifting Scheme

https://cm-bell-labs.github.io/who/wim/papers/spie95.pdf

Predict/update lifting offers an in-place multiresolution construction. Retaining detail coefficients permits reconstruction; discarding them makes a different, lossy representation.

Inspected: PDF text and page image inspected.

## S11 — Nevill-Manning and Witten (1997), SEQUITUR

https://arxiv.org/abs/cs/9709102

Repeated phrases can induce an incremental hierarchical grammar. Use this as a structural reuse hypothesis, not a claim that grammatical compression discovers functional biology.

Inspected: Abstract and author publication site.

## S12 — Bannai et al. (2018), Refining the r-index

https://arxiv.org/abs/1802.05906

Compressed indexing provides exact candidate-location machinery in repetitive collections. Index construction, occurrence output and repetitive posting lists still cost work.

Inspected: Primary abstract.

## S13 — Ashkiani et al. (2018), A Dynamic Hash Table for the GPU

https://arxiv.org/abs/1710.11246

Warp-cooperative work sharing and slab-oriented lookup offer a routing substrate. A safe bulk-built directory is the first experiment; fully concurrent mutation is a separate problem.

Inspected: Primary abstract and HTML.

## S14 — Ashkiani et al. (2017), GPU Multisplit

https://arxiv.org/abs/1701.01189

Programmer-defined bucket partitioning can avoid unnecessary within-bucket sorting. It suggests opcode cohorts and carrier routing; published throughput on other devices is not imported as a V100 result.

Inspected: Primary abstract.

## S15 — Petersen et al. (2022), Deep Differentiable Logic Gate Networks

https://arxiv.org/abs/2210.08277

Continuous relaxations can learn discrete gate networks. Transfer the learned-to-Boolean compilation route, while measuring hardening error separately from task accuracy.

Inspected: Primary abstract and author implementation.

## S16 — Petersen et al. (2024), Convolutional Differentiable Logic Gate Networks

https://arxiv.org/abs/2411.04732

Convolutional logic-gate constructions expand the expressivity of learned Boolean operators. This motivates local learned predicates beyond a fixed motif dictionary.

Inspected: Primary abstract.

## S17 — Anderson et al. (2021), Efficient Parallel Self-Adjusting Computation

https://arxiv.org/abs/2105.06712

Tracking dependencies can reuse unaffected computation after input changes. Genome edits require both value invalidation and repair of any changed routing/partition structure.

Inspected: Primary abstract.

## S18 — Willsey et al. (2021), egg: Fast and Extensible Equality Saturation

https://arxiv.org/abs/2004.03082

E-graphs organize equivalent expressions and rewrite-driven optimization. Use a tiny typed offline rewrite experiment, not another general runtime compiler inside Baseplane.

Inspected: Primary abstract and HTML.

## S19 — Huynh, Knezevic and Patera (2013), Static condensation Reduced Basis Element method

https://numdam.org/articles/10.1051/m2an/2012022/

Interface-level condensed response operators motivate port summaries. The source’s mathematical guarantees concern its numerical problem; they do not establish a genomic response model.

Inspected: Primary article abstract.

## S20 — Nguyen et al. (2023), HyenaDNA

https://arxiv.org/abs/2306.15794

A long-range single-nucleotide sequence-model precedent. It is a scientific/model baseline to understand, not a substitute for Baseplane’s adaptive hardware-native representation question.

Inspected: Primary abstract.

## S21 — Avsec et al. (2021), Enformer

https://www.nature.com/articles/s41592-021-01252-x

Sequence-to-functional-output modeling provides a concrete biological target family. Predictive associations and variant scores do not by themselves establish causal mechanisms.

Inspected: Primary paper abstract.

## S22 — Harris et al., GPU Gems 3, Parallel Prefix Sum (Scan) with CUDA

https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

Scan is a reusable parallel primitive for ranks and composition. Adapt the algorithmic idea to Volta synchronization rather than copying old warp-synchronous assumptions.

Inspected: NVIDIA chapter.
