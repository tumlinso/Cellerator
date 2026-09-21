# Integrated native-foundation qualification

The source pin is recorded with the gate evidence.  Qualification uses the
actual registered tests rather than formula-only substitutes:

* `ce_nf1_h02`: registered custom block identity, duplicate rejection,
  composed/custom parity, predicate exclusion, and host FP32/FP16 routes.
* `ce_nf1_h03`: CUDA grouped n-ary evaluation.
* `ce_nf1_width`: width 1/16/33 arithmetic tail matrix and CUDA response path.
* `ce_nf1_program_preflight`: whole-program metadata admission before any
  accepted launch; later device failure remains nontransactional.
* `ce_nf1_v07`: independent instance/lifetime/readiness and CUDA capture
  refusal; `ce_nf1_d02`: resident-primal response JVP/VJP/second actions.
* `ce_nf1_external_custom_operation`: a consumer-defined CUDA forward launcher
  is registered as a `compiled_block`, executes through `prepared_program_v2`,
  and proves that its declared forward-only contract rejects JVP binding before
  a launch. Its registered `ce_nf1_external_custom_operation_memcheck` uses the
  Compute Sanitizer paired with the configured CUDA compiler.

The standalone `ce_nf1_installed_external_custom_consumer` project builds the
same consumer-defined block with `find_package(Cellerator CONFIG REQUIRED
COMPONENTS native_foundation)` against the pinned install. It must be run from
that configured consumer build, never by compiling Cellerator source into it.

A direct standalone-consumer CTest was accidentally run during local
configuration without the controller lease. It is invalid and excluded from
qualification evidence; the controller foreground lease below is the only
accepted GPU execution.

Forward-only custom blocks are not derivative blocks: `bind_compiled_stage`
rejects requested JVP/VJP/second actions when the registered contract omits
the capability. This is exercised through program preflight rather than
silently substituting a zero derivative. The controller foreground lease
`foreground/bb9f0ce2-8cab-4b17-9b09-3543f6ac9696/lease.json` ran
`ce_nf1_width`, `ce_nf1_d02`, `ce_nf1_h03`, the registered external custom
consumer and its memcheck before benchmarking the installed consumer under
Compute Sanitizer on the qualifying V100. `ce_nf1_v07` remains a separate
sealed owner-bound wrapper; it must be rerun by the root native gate with its
registered bindings and is not claimed by this generic controller run.
