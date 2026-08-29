# Cellerator Scope

## Purpose

Cellerator exists to make biological computation faster by compiling biological organization into accelerator organization.

Its primary concern is not a file format, a model family, a sparse-matrix API, or a high-level workflow. Its concern is the complete path from biological structure to efficient execution:

```text
biological domains and relations
    → semantic execution geometry
    → physical projections
    → planned native or vendor kernels
    → persistent execution order
    → quantitative biological state
```

Performance is the governing criterion. Biological semantics belong in the core when they expose exploitable regularity, enable correct identity and ordering, or permit cheaper execution.

## In Scope

Cellerator owns:

- biological domain, axis, order, geometry, partition, structure, and value-generation identity;
- typed relations between biological domains;
- immutable sparse and hierarchical structure;
- mutable numerical value planes;
- CellPack and CP-BP biological geometry compilation;
- row, feature, module, and nested partition optimization;
- physical projection construction and caching;
- native sparse, masked, block-sparse, dense-fragment, reduction, graph, and sequence-conditioned kernels;
- CP-Math planning, autotuning, prepared execution, epilogues, and backend selection;
- graph-wide order and conversion optimization;
- sparse-to-dense learned transitions that consume native biological structures directly;
- training-oriented forward, transpose, backward, and optimizer primitives;
- precision and quantization policy tied to biological modules or execution blocks;
- single-GPU and multi-GPU execution planning;
- layout-aware sparse transforms, reductions, and biological operators reused by higher layers;
- independent correctness referees, structural validation, and performance instrumentation;
- narrow shared ABI contracts used by Baseplane, CellShard, CelleraTorch, and other adapters.

## Subordinate and Adjacent Repositories

### Baseplane

Baseplane is a subordinate library within the Cellerator conceptual and computational umbrella.

It owns sequence-specialized engineering:

- compact nucleotide representations;
- validity-aware sequence views;
- bit-plane transforms;
- exact and bounded sequence predicates;
- motif and grammar programs;
- masks, events, segments, and sequence-local reductions;
- CPU, SIMD, and CUDA implementations of those primitives.

Cellerator owns their integration with regulatory structure, quantitative state, planning, and downstream execution.

### CellShard

CellShard owns storage and distribution:

- canonical and sharded persistence;
- source ingest and durable publication;
- execution-envelope storage;
- generation and compatibility validation;
- fetch, cache, transport, upload, and delivery to workers;
- storage-oriented partition metadata.

CellShard does not own Cellerator's biological geometry, physical projection semantics, planner, or kernels.

### CelleraTorch

CelleraTorch owns Torch and libtorch adaptation:

- tensor views;
- custom-op registration;
- framework module wrappers;
- explicit conversion at framework boundaries.

It does not own native Cellerator structures, parameters, planning, or reusable math.

## Higher Layers

Model, trajectory, and workflow code may remain in Cellerator when they:

- exercise Cellerator-native biological operations;
- provide reusable numerical policy;
- expose performance-relevant execution structure;
- validate the core on realistic workloads.

They must not duplicate core math, runtime ownership, structure semantics, or planning.

Conventional preprocessing semantics, QC policy, normalization conventions,
workflow ordering, and dataset orchestration belong in downstream BioPrep.
Cellerator supplies only the reusable low-level operations that such a package
composes.

## Out of Scope

Cellerator does not own:

- canonical biological dataset storage;
- generic object-store, HDF5, network, RDMA, or file-service infrastructure;
- a replacement for CUDA, cuBLAS, cuSPARSE, CUB, NCCL, MPI, or UCX;
- a general-purpose tensor framework;
- a Scanpy, AnnData, or notebook-workflow replacement;
- a universal sparse matrix library unrelated to biological structure;
- automatic preservation of every historical API;
- a requirement that one format win for every workload;
- biological ontology for its own sake;
- runtime reconstruction of modules that could have been compiled;
- hidden conversion at framework or storage boundaries.

## Scope Tests

A proposed core feature belongs in Cellerator when most of the following are true:

- it operates on biological domains or relations;
- biological structure changes the efficient implementation;
- it affects order, projection, scheduling, fusion, precision, or communication;
- it can be reused across models or workflows;
- a generic framework would obscure or mishandle the relevant cost;
- its correctness requires biological identity beyond array shape;
- it can reduce bytes, launches, synchronization, or communication.

A proposed feature probably belongs elsewhere when it is primarily:

- canonical storage or transport;
- source parsing;
- framework registration;
- user-interface workflow;
- generic numerical functionality with no biological execution advantage.

## Transitional Exceptions

The current repository contains older layout, preprocessing, model, and runtime surfaces that predate this scope. They may remain while they provide:

- working fallbacks;
- reference implementations;
- tests;
- migration adapters;
- realistic workloads;
- benchmark baselines.

Do not interpret their continued presence as permanent ownership or as evidence for the future core ABI.
