# Baseplane moonshot bootstrap

**Prepared 2 October 2026.** A researched successor program for Baseplane's deferred BitOp roadmap, with 48 experiment cards, 12 implementation families, 10 cross-family compositions and original executable seeds.

**Live-state effect: none.** The remote repository and Todo authority were inspected read-only. The successor native plan has not been applied or natively validated. CUDA seeds have not compiled or run in this environment. Host references, the small learning fixture and package structure have local receipts.

## The architectural idea

A genome need not become a dense vector at every base before it becomes useful context. Try letting sequence produce **composable effects, refinable summaries, source-grounded questions and routable relations**. Rich learned representations remain the destination; bits and machine-native structures decide where and how that representation is formed.

The strongest new candidate represents a region by what it does to incoming state. For a 32-state machine, one warp stores the complete transition function and two regions compose through a shuffle. A continuous cousin composes channel permutations, scales and offsets. Residual hierarchies make coarse representations revisitable; real candidate directories make nonlocal relations discoverable instead of assuming an oracle.

These are competing and composable hypotheses, not a mandated final architecture. The programme also explores learned chunks, Boolean circuit learning, texture response tables, factor joins, possible-state Tensor Core tiles, counterfactual worlds, dynamic precision and more.

## Read

[Recovered intent](docs/00_INTENT.md) distinguishes the user's objective from new proposals. [Repository review](docs/01_REPOSITORY_REVIEW.md) explains the actual substrate, unfinished work and previous lab results. [Architectural search](docs/02_ARCHITECTURAL_SEARCH.md) gives the central mechanisms and equations. [Experiment catalogue](docs/04_EXPERIMENT_CATALOGUE.md) links all 48 implementation-facing cards. [Volta palette](docs/03_VOLTA_PALETTE.md) keeps the design target honest.

For execution, use [the adoption and supersession procedure](docs/05_EXECUTION_AND_SUPERSESSION.md) and [the root handoff](AGENT_HANDOFF.md). The source-native payloads are `machine/baseplane.todo-plan.json` and `machine/cellerator.todo-plan.json`; all 33 old task dispositions are in `machine/old-to-new.json` and CSV. Required policy changes and the signed-run-replacement intent are separate explicit files. [Later qualification](docs/06_LATER_QUALIFICATION.md) is deliberately not scheduled in this workshop.

## Run the supplied host seeds

```sh
python3 tools/check_package.py
./tools/build_host.sh
python3 seed/python/learn_lut.py
```

The reference library includes exact LUT/rank/counter fixtures, finite-state and counted composition, cross-channel affine composition, residual reconstruction and query refinement, global grouping, exact interning, source-support unions, dependency invalidation and finite relation composition. The demonstration connects sequence-derived toy effects to a residual hierarchy and exact source revisits; it is not a learned genome model.

## Compile CUDA later, without launching it

```sh
./tools/compile_cuda.sh
```

Use a configured sm_70-capable compiler, preferably the inspected CUDA 12.9 route. Optional `CUDAHOSTCXX` selects an existing compatible host compiler. The script never chooses a GPU or runs the binary. The future smoke harness requires explicit `--run DEVICE` after resource assignment.

CUDA seeds include LOP3 masks, warp transition composition, counted transitions, monomial composition, lifting, capacity-accounted emission, warp-local equal-key masks, bit-sliced counters, FP16/FP32 WMMA, relation thresholding, DP4A, butterfly mixing and a texture response kernel. Their authored status is not a performance or correctness receipt.

## Evidence and source use

`evidence/observed-state.json` records the remote baseline. `evidence/repository-sources.json` lists inspected paths/ranges. `REFERENCES.md` and `machine/sources.json` contain 22 primary sources and their limited takeaways. All new Baseplane mechanisms and transfers are explicitly proposals, not results borrowed from those sources.

`evidence/validation.json` separates host checks, package checks, unavailable native validation and unavailable CUDA checks. `MANIFEST.sha256` protects the delivered bytes; it is not a signature or a Todo authority.

## Cellerator research ownership

The user's final clarification is implemented throughout the package: Cellerator's scope includes experimental learned mechanisms and ML-like math, not just mature numerical kernels. The Baseplane plan remains the sequence campaign and supersedes the old BitOp schedule. A seven-record companion Cellerator plan owns numerical effect maps, Tensor Core mechanisms, learned routing/hardening and port/factor/counterfactual operators from inception. See [paired research ownership](docs/07_CELLERATOR_RESEARCH_SCOPE.md). Neither plan has been applied, and existing Cellerator ML2 work is not superseded.
