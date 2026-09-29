# Curating evidence without cherry-picking

Select results because they explain the project: a regime-dependent win, an execution capability, a bounded failure that changed a design decision, or an information/cost tradeoff. Use at most two highlighted figures per repository and link the rest. The shortlist in this package is editorial, not an automatic ranking by speedup. Record why every considered candidate was shown or omitted. Keep unfavorable cases in the selected study's full sweep.

## Evidence states

- **Recorded summary:** numbers transcribed from an inspected repository report. Useful in package previews, explicitly not a new run. Some raw artifacts still need portable export.
- **Verified historical:** the implementing agent verified the report/raw artifact identities and methods. The date/source remain historical; no fresh-hardware claim is implied.
- **Fresh measurement:** a local run on the stated code, hardware and command, with durable raw evidence.
- **Pending:** no result exists; render a clearly labeled empty panel, not invented bars or zero measurements.

Public result pages require a local reviewer, validation at a concrete source HEAD, checked command/method/source paths, and reachable evidence. Public charts must never depend on hidden runtime URLs or the user's absolute workstation paths. Export a minimal non-sensitive evidence copy with a manifest of origin and hashes when needed; never modify original receipts. Missing metadata is shown as unknown, not guessed.

## Comparison contract

Record hardware, software/toolchain, build flags, precision/accumulation, dimensions and data construction, current/archived code identity, warmup/repeats, order/randomization, synchronization and memory ownership. Identify exactly which phase is timed: kernel, prepared device pipeline, amortized caller-visible execution, or full lifecycle. Count setup, packing, conversion, transfer, queue/routing and teardown when the claim requires them. Same task, input, numerical tolerance and outputs are required for a performance ratio. A more expensive but different-quality computation is not an equivalent baseline. Lossy routing requires an information/quality measurement.

Never average ratios across unrelated workloads, report only the fastest sample, omit awkward density/reuse points, or plot unlike mechanisms on a single throughput league table. Missing conditions are missing, not zero. For small samples display the available raw values/range and name the interval; min–max is not a confidence interval. Repeated measurements within one host invocation are not independent replications.

A meaningful numerical/correctness failure belongs in a capability/failure panel, **not** as a competitive speed bar. Valid-but-slower candidates can appear in timing plots with their decision explained. Do not convert an entire project's status into one “passed tests” badge.

## Small scope for execution

Reuse existing measured campaigns first. At most one bounded refresh per selected study, with existing workload points; if evidence is already reproducible and current enough, verify/archive instead. Do not rerun every historical sanitizer campaign for a Markdown-only change. After code moves, run affected build/tests and one targeted benchmark smoke before claiming performance continuity. No device clock/power change without the configured owner. Keep profiler runs separate from clean timing runs.

Additional measurement is limited to the optional GlassHelix evidence-discrimination fixture in `benchmarks/GH_CAPABILITY.md`. It is a small wrapper around existing facilities, not model training. Its analytical oracle is prewritten; the agent must call the actual library and distinguish its outputs from reference values.

## Rendering

Figures are generated from checked result JSON with units, provenance and limitations in the caption. Bar axes start at zero. Lines use actual numerical spacing; all relevant points remain visible. Direct labels, markers/line styles and tables make color nonessential. Pending templates contain no numerical marks. README callouts link to the study; the detailed page links to raw evidence and reproduction instructions.

`tools/figures.py` is presentation only. It neither executes programs nor installs dependencies. `tools/check.py --final` rejects unresolved bindings, unreviewed selected studies and absent local evidence. It is a bounded checker, not a claim that a whole software stack is verified.
