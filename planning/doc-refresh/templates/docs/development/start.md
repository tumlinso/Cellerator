# Build, use and validate

Start with one supported path rather than every historical target. Commands below must be qualified against the source version in the [current snapshot](../status/current.md).

{{BUILD_COMMANDS}}

The current host `cellerator` executable forwards an invocation to a selected host compiler, for example `./build-docs-host/cellerator --driver c++ source.cc -o program`. `./build-docs-host/celleratord` validates and prints its architecture-v1 descriptor. These host smokes verify the present passthrough/tooling route; they do not establish complete `.cell` compilation or CUDA runtime support.

## Work on a bounded component

Follow the [source map](source-map.md), inspect its nearest guidance and tests, and keep an example/reference beside the behavior it exercises. A documentation change does not require a full hardware campaign. A source move does require updating build/install/import/include references and rerunning the affected targets.

For measured claims follow [results](../results/index.md). Use assigned resources; record command, source, toolchain, input, precision, outputs, warmup/repeats and timed phases. Keep profiler diagnostics separate from benchmark timings. Preserve the original evidence when rendering new documentation.

## Code and documentation boundaries

Durable design goes in `docs/design/`, practical instructions here, dated summaries in `docs/status/`, and evidence in `docs/results/` linked to original records. Use experiments for unpromoted mechanisms. Generated Todo state and historical notes are not substitute architecture.
