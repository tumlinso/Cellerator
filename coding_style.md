# Cellerator C++ coding style

Cellerator uses compact, explicit numerical and systems code in the tradition
of classic C and Fortran scientific libraries.

**Use the compact form by default. Expose the calculation clearly.** These are
simultaneous requirements. Achieve clarity through precise notation,
mathematical grouping, deliberate layout, and direct program structure.
Additional source must contribute concrete computational or contractual meaning.

Source must expose the algorithm, memory layout, and execution through arrays,
pointers, indices, loops, arithmetic, and deliberate function calls. Use short
mathematical notation to make the calculation recognizable and keep related
work in view.

This governs how programs are understood, independently of performance.
Explicitness means making important operations visible. Concise code must be
backed by a precise account of what it asks the machine to do. Use C++ features
to express useful structure and remove repetition while preserving that
visibility. Keep dependencies and library machinery minimal.

## Authority and protection

Read this document before C++ design, implementation, or style review in
Cellerator, including CUDA C++. It is protected contributor authority.
**Do not change, rewrite, weaken, replace, move, or remove `coding_style.md`,
or weaken its protection in `AGENTS.md`, unless the user explicitly authorizes
that specific change.** A general implementation, cleanup, documentation, or
refactoring request does not grant that authorization. Report conflicts with
other guidance, tooling, or implementation to the user; do not silently revise
these rules to match. Keep the rendered include faithful to this canonical
root text.

Apply these rules within the authorized task. They do not authorize broad
refactoring or changes to biological semantics, numerical contracts, ownership,
or execution authority. Those remain governed by the repository's architecture
and operation-specific contracts.

## Scope and dependencies

These rules cover all authored native production code: public interfaces,
templates, examples, host preparation and planners, resource owners, runtime
machinery, CPU calculations, and CUDA kernels. Host code, cold paths, and
placement in a Cellerator class are not exemptions. Cellerator owns reusable
numerical, storage, layout, execution, derivative, and modeling abstractions;
those abstractions must preserve the same direct computational style.

No STL machinery: no containers, algorithms, smart pointers, type-erased
callables, or equivalent machinery concealed behind another name. This includes
vector, map, set, tuple, variant, string, optional, span, ranges, function, or
generic growing-container replacement. Use visible fields, pointers, counts,
capacities, indices, and explicit initialization. Reuse existing Cellerator owners and views where
their contracts fit; do not duplicate them in a consumer or replace prohibited
STL with another general-purpose framework.

Use the smallest necessary scalar headers: stddef.h, stdint.h, and float.h.
Prefer compiler math and memory builtins. Audit authored transitive includes
as well as direct ones. Necessary platform, CUDA/intrinsic, and vendor API
declarations belong at their concrete implementation or adapter boundary;
foreign API-mandated types may appear only where that API requires them.
These exceptions do not license STL-backed internal planning or storage.
Development-only tests, tooling, and measurement infrastructure may use ordinary
library facilities. Third-party implementations follow their own rules. Keep
those dependencies out of the authored core except at the stated boundaries.

Existing noncompliant source is migration debt, not precedent or an exception.
New and reworked code must comply. Fix the affected computation and necessary
callers within the authorized task; unrelated repository-wide cleanup is not
a prerequisite for using an existing interface or completing scoped work.

## Expressions and mathematical notation

- Use `?:` for value selection, including initialization, arguments, and
  returns. Use a Boolean expression directly when it already gives the result:
  `return i < n;`.
- Express short piecewise definitions as linear ternary chains. Align cases
  that benefit from separate lines. Express distinct actions and sequencing
  with direct control flow, keeping the branch structure visible.
- Keep mathematical expressions whole: `y[i] = a*x[i] + b*y[i];`. Introduce a
  temporary for reuse, a meaningful intermediate quantity, an explicit
  conversion, numerical staging, or a dependency that benefits from a name.
- Do not expand straightforward calculations into named arithmetic steps,
  verbose local names, repetitive statements, or forwarding helpers. Introduce
  intermediate quantities for reuse, mathematical significance, or required
  evaluation semantics. Keep the calculation recognizable as a whole.
- Use conventional arithmetic and compound assignments where their semantics
  fit. Each expression must have traceable effects and defined sequencing.
  Parenthesize grouping that would otherwise require deciphering precedence.

These examples show the default forms:

```cpp
const int step = reverse ? -1 : 1;
int sign(int k) { return k < 0 ? -1 : k > 0 ? 1 : 0; }
```

Condensing an expression must preserve its numerical and language semantics:
types, conversions, rounding where required, evaluation dependencies, aliasing,
and lifetimes. Retain useful temporaries rather than repeat a computation.
A ternary evaluates its selected arm, while both arms participate in determining
its result type and value category. It is no promise of branchless execution.

## Names, declarations, and types

- Use short names tied to established notation: `i`, `j`, `k`, `m`, `n`, `lda`,
  `dt`, `r`, and `p` are precise in a clear local context. Give broader interfaces
  enough naming context to stand alone.
- Group related scalar declarations, such as `int m = a.rows, n = a.cols;`.
  Keep pointer and reference declarations individually clear. Group statements
  by their role in the calculation.
- Show consequential numeric types, index widths, and conversions. Keep storage,
  multiply, accumulation, and output types explicit where they differ, following
  the numerical policy and the operation's contract.
- Use `auto`, `auto &`, and `auto *` deliberately. Make the distinction between
  a value, an alias, and a pointer apparent, especially where a value declaration
  would copy owned storage.
- For known arithmetic types, use short conversions such as `double(n)` and
  `float(x)`. Use named casts where pointer interpretation, qualifiers, or class
  relationships deserve attention.
- Use short namespace aliases and useful type aliases to simplify repeated
  notation. Their meaning and numerical or ownership consequences must remain
  easy to find. Scalar declarations do not authorize broad runtime headers.

## Control flow and function boundaries

- Keep simple guards, loops, and short functions compact, using one-line forms
  as the default. Omit braces for unambiguous single-statement bodies; use braces
  for multiple statements or grouping that needs to be visible. Preserve
  unambiguous `else` association.
- Use early returns and `continue` where they keep the ordinary path flat and
  reduce the state a reader must remember.
- Group tightly coupled short statements, such as `x += dx; y += dy;`. Use
  conventional pointer idioms such as `*q++ = *p++;` for pointer advancement.
  Preserve valid extents and the aliasing contract.
- Keep a coherent algorithm together. There is no function-length quota.
  Extract functions for useful operations, reusable computations, or distinct
  contracts. Split at meaningful behavioral boundaries rather than to satisfy
  a line count.
- Ordinary labels and `goto` may expose phases, retries, cleanup, or exits
  from nested loops. Computed goto is a documented compiler extension confined
  to compiled implementation functions. Keep targets local and obey object
  initialization, lifetime, and destruction rules.

For example:

```cpp
if (!n) return;
for (int i = 0; i < n; ++i) y[i] += a*x[i];
```

## Data, ownership, and abstraction

- Use plain structs, direct data access, and free functions for plain data
  and computations. Give classes responsibility for ownership or real
  invariants. Getters, setters, factories, and inheritance need a concrete
  purpose.
- Keep owners and borrowed views distinct. Make pointer stability, extents,
  strides, layout, and lifetimes understandable from the interface and code.
  Allocation, copying, conversion, transfer, synchronization, and runtime
  dispatch must be apparent through clear owners and deliberate calls.
- Separate explicit resource provisioning from numerical work. Sizing,
  validation, numerical preparation, binding, and evaluation use supplied
  storage. Cheap stack descriptors and borrowed views are permitted; they do
  not acquire owners, materialize data, or allocate bookkeeping.
- Establish contracts at boundaries and let internal routines rely on those
  documented preconditions. Recheck facts when mutation or a new boundary can
  invalidate them. Preserve useful compiler-visible information about valid
  types, ranges, alignment, and aliasing; assert only guarantees the contract
  actually provides.
- Use small templates, callable structs, local lambdas, and `if constexpr` to
  express genuine common structure and useful specialization. Introduce
  runtime polymorphism, registration, or policy machinery only for a concrete
  in-scope computational contract that simpler direct code cannot express.
  Prefer typed function pointers with explicit caller state at deliberate
  operation boundaries; keep type erasure and dispatch out of element loops.
- Allow deliberate duplication when traversal, numerical treatment, or
  ownership genuinely differs. Share algorithms whose structure is common;
  keep distinct algorithms recognizable and account for generated code size.
- Judge an abstraction by the total source and reasoning required to understand
  its use and implementation. Helpers, wrappers, classes, policies, registration
  systems, and dependencies must remove identifiable complexity or enforce a
  concrete contract. Count everything the reader must inspect to understand the
  computation. Moving complexity behind a short call does not discharge that
  obligation. Appeals to standard practice, extensibility, or readability must
  identify the specific benefit provided here.

## Provisioning, allocation, and reuse

Only explicit resource provisioning acquires, grows, or releases backing
storage. A reserve request that fits existing capacity, alignment, and
placement must allocate, copy, and transfer nothing. Numerical operations
report insufficient capacity and required storage instead of silently growing
buffers. Requirements must include metadata, prepared topology, diagnostics,
ownership bookkeeping, and scratch as well as numerical outputs. Data-dependent
sizes may use a counting pass, a sufficient bound, or capacity reporting;
choose the least costly strategy suitable for the operation.

Keep data/results, reusable prepared state, and transient scratch lifetimes
explicit. Each allocation has one releasing owner; views borrow storage.
Bulk owners and lightweight cell/feature views must avoid per-cell or per-entry
allocation. Required initialization, copying, packing, transfer, and waits are
deliberate operations. Retain storage through its final asynchronous consumer;
resetting a workspace or publishing a generation does not prove completion.

Repeated evaluation within provisioned capacities performs zero heap/device
allocations or backing-storage growth, including hidden metadata and control
blocks. Compatible smaller inputs reuse capacity. Changed numerical values in
an unchanged representation within capacity do not justify reallocating. Reuse
prepared topology while its documented dependencies remain valid; rebuild only
affected state when data-derived support, membership, or eligibility changes.
A pool does not exempt temporary churn, needless copies, or excessive retention.
Account for allocation count/bytes, peak live and retained bytes, and material
copy/transfer costs across the complete operation lifecycle.

Use the simplest existing owner and workspace facilities that satisfy the
operation. A new allocator, pool, cache, memory planner, or ownership framework
requires a demonstrated in-scope need; it is not a routine prerequisite for
adding a calculation. Caller-owned bounded workspace may be subdivided without
acquiring backing storage. Reuse follows the caller's concurrency and completion
contract, not an implicit global synchronization or background scheduler.

## Hot paths and compilation cost

**Prefer compile-time optimization and useful specialization over recurring
runtime overhead in hot paths; accept longer builds to obtain better execution.**

Spend compilation effort to remove recurring runtime work. Use available static
information and useful specialization through templates, static dispatch,
`if constexpr`, and compile-time knowledge of types, layouts, extents, and
alignment. Separately compile ordinary host calculations and common
instantiations; keep public interfaces small and implementation-only edits from
recompiling unrelated consumers. Keep templates and device helpers visible
where actual specialization, CUDA compilation, or explicit consumer
customization requires it. Compilation expense alone does not justify recurring
hot-path runtime overhead. Template machinery must remove identifiable work or
repetition; account for code growth and consumer rebuild cost.

Assess the resulting execution and complete workload costs. Code growth,
instruction-cache pressure, register pressure, spills, memory traffic, and
retained workspace are runtime costs too. Concision carries no permission to
introduce hidden allocation, copying, conversions, repeated work, or unnecessary
specialization. Preserve the required numerical semantics and justify
performance claims with appropriate measurements through the existing tools.
Formatting changes alone do not require a benchmark campaign.

## Whitespace and comments

Use spaces and line breaks to expose the calculation: `a*x + b*y` separates
terms from factors, and aligned cases or nested loops can reveal structure.
Keep related operations together and use blank lines for meaningful phases.

Comments should supply information absent from the syntax: layouts, units,
identity, ordering, extents, aliasing, ownership, lifetimes, invariants, numerical
semantics, and reasons for a representation or synchronization boundary. Record
measured reasons for non-obvious optimizations with their evidence. Avoid
narrating obvious operations or declaring code fast without supporting results.

## Review and acceptance

Review for avoidable verbosity, boilerplate, fragmentation, and obscured
computation independently of correctness and performance. A verbosity finding
should show the existing passage and a compact equivalent, identifying any
semantic, execution, or cost differences. A compliance assessment must examine whether
the direct calculation remains visible throughout the implementation.

Reviewers must actively seek direct expressions of the computation and identify
unnecessary expansion. A claim of clarity must identify the operation or
relationship being exposed. Correct behavior and the absence of a measured
performance regression do not establish style compliance.

Use focused allocation-sensitive checks when changing ownership, preparation,
or workspace behavior: provision once, repeat with changed values and compatible
smaller inputs, and verify zero further allocations/growth and no unnecessary
copies. Include first-use provisioning costs separately; inspect metadata and
retention as well as backing allocator calls. Vendor calls and pool use do not
establish this guarantee by themselves. Do not create a benchmark campaign or
new allocator instrumentation framework for prose or formatting changes.

The scoped source check in the implementation style guide is a lexical guard,
not proof of allocation behavior, numerical clarity, or full policy compliance.
Review the whole affected calculation and its helpers. Do not weaken checks,
hide forbidden machinery behind aliases/macros, or expand legacy allowances to
make new code pass.
