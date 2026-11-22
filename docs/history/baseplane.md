# Baseplane and Moonshot source history

The [native sequence guide](../development/sequence.md) describes current
Cellerator source ownership. These revisions retain the original code,
benchmark fixtures, provider interfaces, and source-bound receipts at their
original paths.

| Material | Preserved source |
| --- | --- |
| Baseplane exact sequence core, tests, and benchmarks | [Baseplane at `8edb8e5`](https://github.com/tumlinso/Baseplane/tree/8edb8e5ed67c5ba522b4e33952becdc843f08c68) |
| Ordered summaries, exact source revisit, edit repair, candidate verification, and learned routing | [Baseplane Moonshot](https://github.com/tumlinso/Baseplane/tree/8edb8e5ed67c5ba522b4e33952becdc843f08c68/experiments/moonshot), [hierarchy checks](https://github.com/tumlinso/Baseplane/tree/8edb8e5ed67c5ba522b4e33952becdc843f08c68/tests/substrate/hierarchy), and [incremental checks](https://github.com/tumlinso/Baseplane/tree/8edb8e5ed67c5ba522b4e33952becdc843f08c68/tests/substrate/incremental) |
| Baseplane CUDA lab, including the CarryFold comparison and rejected candidates | [CUDA lab selection report](https://github.com/tumlinso/Baseplane/blob/8edb8e5ed67c5ba522b4e33952becdc843f08c68/experiments/cuda_lab/results/selection/report.md) |
| Cellerator experimental effects, mechanisms, learning, and provider receipts | [Former `experiments/baseplane_moonshot`](https://github.com/tumlinso/Cellerator/tree/2c2c222f36c3aa1f7f8bb22d97031002c6420978/experiments/baseplane_moonshot) |
| Cellerator Moonshot derivative proofs, state and trajectory references, rewrite witnesses, and MMA checks | [Former `experiments/moonshot-parallel-v1`](https://github.com/tumlinso/Cellerator/tree/2c2c222f36c3aa1f7f8bb22d97031002c6420978/experiments/moonshot-parallel-v1) |

The Cellerator experiment trees at `2c2c222f36c3aa1f7f8bb22d97031002c6420978`
match the trees at the absorption review base
`afd2ad00615ddece363877bfb48c194f4de99af2`. That review base also retains the
original `docs/baseplane_integration.qmd`, the standalone
`tests/biological_abi/baseplane_cpu_consumer`, and the former
`tests/substrate/{install,frontier}/dependencies.sha256` manifests. For example,
`git show afd2ad00615ddece363877bfb48c194f4de99af2:tests/substrate/install/dependencies.sha256`
retrieves the original install manifest.

Use the revision and invocation recorded with each receipt to reproduce the
historical workload. Cellerator’s maintained SDK exposes the native sequence and
effects operations through their current components.
