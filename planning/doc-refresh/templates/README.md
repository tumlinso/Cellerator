# Cellerator

**What if the organization of a biological system could determine the program that computes it?**

A cell's measurements may arrive as a large matrix, but that matrix is not the whole computational story. The same relations can recur across cells and time points; topology may remain useful while expression, activity and learned weights change. Cellerator explores whether those regularities can be discovered once and compiled into work that an accelerator performs well.

It is a **compiler and execution substrate for biologically structured computation**. Rather than choosing one universal sparse format, it turns useful structure into execution order, physical layouts and generated or assembled kernels. The central question is not just how to run a matrix operation faster, but whether the biological organization lets us formulate a better computation in the first place.

## The design in one pass

```text
system structure and workload
    → discover reusable organization
    → compile geometry and executable alternatives
    → bind current values and run
```

Structure and value have different lifetimes. A new cell state need not force the system to rediscover a reusable relation. The longer-term design also allows state-dependent gates to activate or reorganize prepared computation. Those gates are discovered before execution; their outcomes depend on runtime state. The full adaptive system is an intent, not a blanket claim about every current path.

Biology and hardware are considered together: repeated supports, order and modular structure matter when they can become cheaper memory access, reusable work, or a better kernel arrangement. A vendor library remains the right answer when it wins the complete comparison.

## What exists now

{{README_STATUS}}

## Evidence, not just a design

{{README_RESULTS}}

The [results index](docs/results/index.md) states which studies have been checked and promoted. Each published study will show its workload, costs, comparison and limitations.

## Where to start

Read the [design overview](docs/design/overview.md), then follow the [source map](docs/development/source-map.md) or [minimal build/use guide](docs/development/start.md). The [current snapshot](docs/status/current.md) separates existing compiler/runtime facilities, experimental candidates and work still planned.

Cellerator provides the numerical execution beneath [GlassHelix](https://github.com/tumlinso/GlassHelix)'s questions about dynamics and mechanism. [Baseplane](https://github.com/tumlinso/Baseplane) develops sequence-grounded representations and shares the numerical boundary rather than becoming a second general runtime. See [how the projects fit together](docs/design/program.md).

This is active research software. Its ambition is broader than its measured execution envelopes.
