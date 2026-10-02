# Primary research and hardware references

Accessed 2 October 2026. The design is a synthesis, not a claim of novelty priority. Hardware facts are taken from NVIDIA's manuals; model-family precedents from papers/author-published proceedings. The package does not reproduce full papers. No PDF text or figures were needed for this review.

| ID | Source | Used for / boundary |
|---|---|---|
| R1 | NVIDIA, Volta Tuning Guide. https://docs.nvidia.com/cuda/volta-tuning-guide/ | SM70 resources, independent scheduling and instruction/memory design. No unmeasured application throughput inferred. |
| R2 | NVIDIA, PTX ISA 8.7, especially §9.7.14.5.1. https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html | Exact four-product m8n8k4 operand/accumulator mapping and target support. PTX is not a promise of one physical instruction or portable WMMA layout. |
| R3 | NVIDIA, CUDA C++ Programming Guide 12.8.0. https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html | WMMA participation/alignment, warp collectives, synchronization and memory contracts. |
| R4 | NVIDIA, CUDA Toolkit 13.0 release notes. https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html | Volta offline compilation/library support removal; use existing CUDA12.x for SM70 prototype. |
| R5 | Novikov et al., Tensorizing Neural Networks, 2015. https://arxiv.org/abs/1509.06569 | Tensor-structured model parameterization. Not proof that an arbitrary sparse operator can be reshaped into a cheap contraction. |
| R6 | Dao et al., Monarch: Expressive Structured Matrices for Efficient and Accurate Training, 2022. https://arxiv.org/abs/2204.00595 | Structured block factors as useful learned models. Our simple LXR family is not the Monarch construction. |
| R7 | Ye et al., SparseTIR, 2022. https://arxiv.org/abs/2207.04606 | Composable sparse representations and transformation schedules, rather than a universal format. |
| R8 | Lu et al., DASP, SC23 official paper page. https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap249.html | Counterexample to a blanket claim that Tensor Cores cannot serve SpMV; not V100-specific evidence for this proposal. |
| R9 | Chen, Goodfellow and Shlens, Net2Net, 2015. https://arxiv.org/abs/1511.05641 | Function-preserving network transformation precedent, not biological actor identification. |
| R10 | Wei et al., Network Morphism, ICML 2016. https://proceedings.mlr.press/v48/wei16.html | Preserving behavior while changing model structure. Does not supply our general nonlinear extraction map. |
| R11 | Evci et al., Rigging the Lottery: Making All Tickets Winners, ICML 2020. https://proceedings.mlr.press/v119/evci20a.html | Learning sparse connectivity dynamically. Not a theorem that support patterns identify mechanisms. |
| R12 | Neil et al., Delta Networks for Optimized Recurrent Network Computation, ICML 2017. https://proceedings.mlr.press/v70/neil17a.html | Temporal change-driven computation. Our ledger reference is a bounded fixed-linear-map construction. |
| R13 | Blondel et al., Polynomial Networks and Factorization Machines, ICML 2016. https://proceedings.mlr.press/v48/blondel16.html | Structured polynomial interactions; distinct from the matrix-polynomial family proposed here. |
| R14 | Ootomo and Yokota, Recovering single precision accuracy from Tensor Cores, 2022. https://arxiv.org/abs/2203.03341 | Precision-decomposition/correction precedent. Reported later-device performance is not transferred to V100. |
| R15 | NVIDIA, CUDA 12.8 Best Practices, texture memory; CUDA 9.2 Programming Guide, texture filtering. https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/ ; https://docs.nvidia.com/cuda/archive/9.2/cuda-c-programming-guide/ | Hardware lookup/interpolation as an optional rate-law realization. Low-precision coordinates/interpolation and immutable texture lifetime matter. |
| R16 | Ootomo and Yokota, Reducing Shared Memory Footprint to Leverage High Throughput on Tensor Cores and Its Flexible API Extension Library, 2023. https://arxiv.org/abs/2308.15152 | Motivation for investigating fragment-level communication after a correct shared-memory reference; not authority to guess WMMA register layouts. |

## Evidence classification

The CPU algebra in this package verifies particular identities and reference implementations. The CUDA source is independently authored from the documented interfaces and remains uncompiled here. A primary source explaining a hardware feature is not evidence that this implementation uses it correctly or runs faster. The live project paths in PROJECT_REVIEW.md establish existing integration opportunities, not performance of the new mechanisms.

The proposed Sylvester-flow, port-coordinate transformation, quotient witness, delta bound and adjoint equations can be checked directly from the displayed mathematics; they are not attributed as empirical findings of these papers.
