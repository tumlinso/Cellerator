# Cellerator — adaptive NF1 completion

Pair **NF1A-20260916-v1**. This is a manual successor bootstrap and controller handoff, not an implementation or a launch. It adds nine local records and seven lanes; the paired program adds 18 records and 14 lanes. The 52 observed completed NF1 records are retained. The 120 observed unfinished records are mapped explicitly, not silently marked successful.

Read `CHARTER.md` and the relevant outcome sheet for execution. Read `ARCHITECTURAL_INTENT.md` when the architectural boundary matters, `SOURCE_REVIEW.md` for evidence, and `RUNTIME_AND_BOOTSTRAP.md` for manual import and interruption handling. Do not load the entire package into every child.

`machine/native-plan.json` is the native schema-3 input template. The host preview binds only the verified Python executable and external execution-bindings path, then saves the exact reviewed ProposalEnvelope. Machine sidecars explain requirements, conditional choices, profiles and old-work mapping; they do not create another authority.

Before launch: validate the package, preview/apply/verify both additive imports, and retain the receipts. The controller starts with the two ADOPT outcomes; no implementation worker is dispatched until both imports and old-work preservation/retirement are verified. A manual import never resumes the old frontier.

Local construction checks are reported in `evidence/local-validation.json`. Exact native on-host validation, runtime/gate binding and device execution are separate facts. An empty test binding deliberately fails; these files contain no synthetic pass for an unimplemented capability.

Local run: `CE-NF1A-RUN-V1`. Initial task: `CE-NF1A-ADOPT`. Ordinary coordinator: `CE-NF1A-CONTROL`. Aggregate `CE-NF1A-0000` is last, not the coordinator seat.
