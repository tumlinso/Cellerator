# Build, use and validate

Start with one supported path rather than every historical target. Commands below must be qualified against the source version in the [current snapshot](../status/current.md).

{{BUILD_COMMANDS}}

## Work on a bounded component

Follow the [source map](source-map.md), inspect its nearest guidance and tests, and keep an example/reference beside the behavior it exercises. A documentation change does not require a full hardware campaign. A source move does require updating build/install/import/include references and rerunning the affected targets.

For measured claims follow [results](../results/index.md). Use assigned resources; record command, source, toolchain, input, precision, outputs, warmup/repeats and timed phases. Keep profiler diagnostics separate from benchmark timings. Preserve the original evidence when rendering new documentation.

## Code and documentation boundaries

Durable design goes in `docs/design/`, practical instructions here, dated summaries in `docs/status/`, and evidence in `docs/results/` linked to original records. Use experiments for unpromoted mechanisms. Generated Todo state and historical notes are not substitute architecture.
