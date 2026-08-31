# Experimental alternatives and promotion gates

Settled architecture is not re-litigated by these gates. Each gate compares a complete baseline with a replaceable candidate. The candidate may end as `evaluated_not_promoted`; the fallback remains a first-class supported path. No implementation agent may silently promote a candidate merely because it is novel or locally faster.

| Gate | Question | Baseline | Candidate | Promotion rule | Fallback |
| --- | --- | --- | --- | --- | --- |
| JBC-PG01 | Flat explicit composition DAG versus induced execution grammar | CS-JBC-G explicit grammar/flat-basis fallback | CS-JBC-IG induced grammar | Exact derivations and positive amortized complete-cost benefit over the flat DAG across the declared reuse horizon. | Retain explicit grammar and flat basis; store negative grammar evidence. |
| JBC-PG02 | One global basis versus multiple workload-family bases | Single greedy basis | Multiple basis portfolio | Portfolio is nondominated and amortized savings exceed additional storage/build/maintenance. | Single basis or no-basis per workload family. |
| JBC-PG03 | Disjoint versus bounded overlapping atoms/bases | Exactly disjoint proposals/basis | Bounded overlap | Complete cost improves with certified unique contribution ownership and bounded duplication. | Zero-overlap path, which must be equivalent to disjoint baseline. |
| JBC-PG04 | Seed-expand versus spectral co-clustering | Deterministic seed-and-expand biclustering | Spectral co-clustering | Better certified atom utility after search amortization on more than planted fixtures. | Seed-expand provider. |
| JBC-PG05 | Bounded typed motifs versus frequent-fragment mining | Hand-bounded typed motif templates | Frequent typed-fragment miner | Discovers profitable atoms missed by templates within explicit budgets. | Bounded motif library or no motif atoms. |
| JBC-PG06 | No factor/topic atoms versus factor-derived proposals | Support/trace/semantic atom providers only | External or internal factor/topic proposals | Adds nondominated certified atoms beyond existing providers. | Retain factor evidence as nonselected proposals. |
| JBC-PG07 | Ordinary trajectory window caching versus prefix/delta materialization | Independent or window-cached states | Prefix, branch and delta atoms | Positive amortized benefit on real/planted trajectory families and collapse under trajectory-null disruption. | State/window atoms without prefix/delta persistence. |
| JBC-PG08 | Explicit contiguous assembly versus direct multi-extent Cellerator execution | CE-JBC-M04/M05 assembly | CE-JBC-M06/M07 direct multi-extent | Complete cost wins in a defined extent/operation region with no hidden full assembly. | Explicit profiler-visible assembly. |
| JBC-PG09 | Per-operation projections versus generalized cross-operation view families | Specialized forward/transpose/contraction/segment views | Generalized operation-polymorphic family | Global graph-family cost wins for a declared operation mixture. | Keep shared semantic support identity with specialized physical views. |
| JBC-PG10 | One-pass external costs versus bounded exchange/column generation | CE-JBC-C04 one-pass frontier | CE-JBC-C05/C06 exchange/pricing oracle | Material global improvement beyond one-pass under strict budget and reproducible calibration. | One-pass external-cost frontier. |
| JBC-PG11 | Synchronous exact reads versus asynchronous multi-range I/O | CS-JBC-RT06 | CS-JBC-RT07/RT08 | Improves complete atom delivery without excessive memory or instability. | Synchronous exact reads and explicit prefetch. |
| JBC-PG12 | Size/LRU-like residency versus reconstruction-aware residency | Simple bounded baseline | CS-JBC-RT18 | Reduces total reconstruction/I/O/compute under stable traces without pathological starvation. | Simple deterministic policy. |
| JBC-PG13 | Raw/index baseline versus CPU/GPU atom-aware compression | CS-JBC-ST26 raw/delta/bitpack/RLE | CS-JBC-ST27/ST28 | I/O-to-kernel complete cost wins for a specific object species/storage tier. | Per-object raw or simple index encoding. |
| JBC-PG14 | One process versus per-NUMA-node or per-GPU runtime | Current in-process executor model | Logical-node/process alternatives in CS-JBC-RT22 | Select per deployment profile; no one process model enters portable schedule identity. | In-process library mode. |
| JBC-PG15 | Relearned atom evidence versus cross-dataset template warm start | Rebuild evidence/atoms from target dataset | Transferred strategy/template proposals | Warm start reduces compile cost without reducing exact quality and is revalidated on target data. | Target-dataset discovery. |
| JBC-PG16 | Demand/sequential prefetch versus trajectory-predictive prefetch | Demand-only or simple sequential | CS-JBC-TR11 | Positive complete-cost benefit under supplied trajectory probabilities and valid null behavior. | Demand/sequential prefetch. |
| JBC-PG17 | CPU atom linker versus GPU-assisted linking | CPU transforms plus explicit H2D | CS-JBC-ST29 | Wins when result remains GPU-resident without displacing more valuable compute. | CPU linker. |
| JBC-PG18 | No superatom versus promoted superatom | Repeated explicit composition | CS-JBC-SA promotion | Positive amortized complete benefit with retained parent lineage. | Composition DAG only. |

## JBC-PG01 — Flat explicit composition DAG versus induced execution grammar

**Baseline:** CS-JBC-G explicit grammar/flat-basis fallback

**Candidate:** CS-JBC-IG induced grammar

**Required metrics:** induction/matching time, grammar bytes, assembly avoided, transforms, I/O, end-to-end graph-family cost

**Promotion criterion:** Exact derivations and positive amortized complete-cost benefit over the flat DAG across the declared reuse horizon.

**Fallback / valid negative result:** Retain explicit grammar and flat basis; store negative grammar evidence.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG02 — One global basis versus multiple workload-family bases

**Baseline:** Single greedy basis

**Candidate:** Multiple basis portfolio

**Required metrics:** storage amplification, graph coverage, assembly, reuse, mutation/invalidation, complete cost

**Promotion criterion:** Portfolio is nondominated and amortized savings exceed additional storage/build/maintenance.

**Fallback / valid negative result:** Single basis or no-basis per workload family.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG03 — Disjoint versus bounded overlapping atoms/bases

**Baseline:** Exactly disjoint proposals/basis

**Candidate:** Bounded overlap

**Required metrics:** occupancy/reuse, duplicated bytes, input replication, gradient reconciliation, storage, exact ownership

**Promotion criterion:** Complete cost improves with certified unique contribution ownership and bounded duplication.

**Fallback / valid negative result:** Zero-overlap path, which must be equivalent to disjoint baseline.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG04 — Seed-expand versus spectral co-clustering

**Baseline:** Deterministic seed-and-expand biclustering

**Candidate:** Spectral co-clustering

**Required metrics:** search memory/time, candidate quality, exact useful support, residual, downstream complete cost

**Promotion criterion:** Better certified atom utility after search amortization on more than planted fixtures.

**Fallback / valid negative result:** Seed-expand provider.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG05 — Bounded typed motifs versus frequent-fragment mining

**Baseline:** Hand-bounded typed motif templates

**Candidate:** Frequent typed-fragment miner

**Required metrics:** search complexity, exact recurrence, reusable execution benefit, null false positives

**Promotion criterion:** Discovers profitable atoms missed by templates within explicit budgets.

**Fallback / valid negative result:** Bounded motif library or no motif atoms.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG06 — No factor/topic atoms versus factor-derived proposals

**Baseline:** Support/trace/semantic atom providers only

**Candidate:** External or internal factor/topic proposals

**Required metrics:** exact candidate utility, storage, overlap, cross-operation reuse, null comparison

**Promotion criterion:** Adds nondominated certified atoms beyond existing providers.

**Fallback / valid negative result:** Retain factor evidence as nonselected proposals.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG07 — Ordinary trajectory window caching versus prefix/delta materialization

**Baseline:** Independent or window-cached states

**Candidate:** Prefix, branch and delta atoms

**Required metrics:** delta bytes, prefix reuse, random access, reconstruction, invalidation, total graph cost

**Promotion criterion:** Positive amortized benefit on real/planted trajectory families and collapse under trajectory-null disruption.

**Fallback / valid negative result:** State/window atoms without prefix/delta persistence.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG08 — Explicit contiguous assembly versus direct multi-extent Cellerator execution

**Baseline:** CE-JBC-M04/M05 assembly

**Candidate:** CE-JBC-M06/M07 direct multi-extent

**Required metrics:** assembly bytes/time, kernel efficiency, TLB/cache, launches, memory, reuse break-even

**Promotion criterion:** Complete cost wins in a defined extent/operation region with no hidden full assembly.

**Fallback / valid negative result:** Explicit profiler-visible assembly.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG09 — Per-operation projections versus generalized cross-operation view families

**Baseline:** Specialized forward/transpose/contraction/segment views

**Candidate:** Generalized operation-polymorphic family

**Required metrics:** local kernel loss, storage, preparation, transforms, canonicalization, cross-operation reuse

**Promotion criterion:** Global graph-family cost wins for a declared operation mixture.

**Fallback / valid negative result:** Keep shared semantic support identity with specialized physical views.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG10 — One-pass external costs versus bounded exchange/column generation

**Baseline:** CE-JBC-C04 one-pass frontier

**Candidate:** CE-JBC-C05/C06 exchange/pricing oracle

**Required metrics:** compile time, candidates added, objective improvement, prediction error, stability

**Promotion criterion:** Material global improvement beyond one-pass under strict budget and reproducible calibration.

**Fallback / valid negative result:** One-pass external-cost frontier.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG11 — Synchronous exact reads versus asynchronous multi-range I/O

**Baseline:** CS-JBC-RT06

**Candidate:** CS-JBC-RT07/RT08

**Required metrics:** /mnt/block throughput, latency, CPU, queue memory, range count, read amplification

**Promotion criterion:** Improves complete atom delivery without excessive memory or instability.

**Fallback / valid negative result:** Synchronous exact reads and explicit prefetch.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG12 — Size/LRU-like residency versus reconstruction-aware residency

**Baseline:** Simple bounded baseline

**Candidate:** CS-JBC-RT18

**Required metrics:** hit rate, rematerialization, stalls, memory, eviction regret, complete graph cost

**Promotion criterion:** Reduces total reconstruction/I/O/compute under stable traces without pathological starvation.

**Fallback / valid negative result:** Simple deterministic policy.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG13 — Raw/index baseline versus CPU/GPU atom-aware compression

**Baseline:** CS-JBC-ST26 raw/delta/bitpack/RLE

**Candidate:** CS-JBC-ST27/ST28

**Required metrics:** stored/read bytes, decode, scratch, H2D, direct consumption, total latency

**Promotion criterion:** I/O-to-kernel complete cost wins for a specific object species/storage tier.

**Fallback / valid negative result:** Per-object raw or simple index encoding.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG14 — One process versus per-NUMA-node or per-GPU runtime

**Baseline:** Current in-process executor model

**Candidate:** Logical-node/process alternatives in CS-JBC-RT22

**Required metrics:** control overhead, NUMA locality, NCCL/CUDA Graph behavior, failures, memory, throughput

**Promotion criterion:** Select per deployment profile; no one process model enters portable schedule identity.

**Fallback / valid negative result:** In-process library mode.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG15 — Relearned atom evidence versus cross-dataset template warm start

**Baseline:** Rebuild evidence/atoms from target dataset

**Candidate:** Transferred strategy/template proposals

**Required metrics:** compile time, candidate recall, exact acceptance, false reuse, end-to-end benefit

**Promotion criterion:** Warm start reduces compile cost without reducing exact quality and is revalidated on target data.

**Fallback / valid negative result:** Target-dataset discovery.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG16 — Demand/sequential prefetch versus trajectory-predictive prefetch

**Baseline:** Demand-only or simple sequential

**Candidate:** CS-JBC-TR11

**Required metrics:** hit rate, wrong-prefetch bytes, memory, stalls, transition uncertainty

**Promotion criterion:** Positive complete-cost benefit under supplied trajectory probabilities and valid null behavior.

**Fallback / valid negative result:** Demand/sequential prefetch.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG17 — CPU atom linker versus GPU-assisted linking

**Baseline:** CPU transforms plus explicit H2D

**Candidate:** CS-JBC-ST29

**Required metrics:** link time, host bytes, GPU occupancy, contention, downstream residency and graph cost

**Promotion criterion:** Wins when result remains GPU-resident without displacing more valuable compute.

**Fallback / valid negative result:** CPU linker.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.

## JBC-PG18 — No superatom versus promoted superatom

**Baseline:** Repeated explicit composition

**Candidate:** CS-JBC-SA promotion

**Required metrics:** assembly avoided, direct execution, storage rent, invalidation, graph-family reuse

**Promotion criterion:** Positive amortized complete benefit with retained parent lineage.

**Fallback / valid negative result:** Composition DAG only.

**Required evidence record:** exact source commits, build/provider identities, dataset/fixture identity, cold and steady-state costs, memory/storage bytes, reuse horizon, numerical result, confidence/variance, and explicit disposition.
