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
- Use narrowly scoped declarations such as `using std::sqrt;`, short namespace
  aliases, and useful type aliases to simplify repeated notation. Their meaning
  and numerical or ownership consequences should remain easy to find.

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
- Allow a forward `goto` to cleanup in low-level C interoperability code when
  it makes resource release clearer. Keep this exception narrow and preserve
  initialization, lifetime, and destruction rules. Use RAII owners where they
  express resource handling clearly.

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
- Separate storage management from numerical work where useful. Pass arrays,
  views, and explicit scratch storage into algorithms so their working set and
  allocation behavior remain understandable. Choose workspace size and lifetime
  deliberately.
- Establish contracts at boundaries and let internal routines rely on those
  documented preconditions. Recheck facts when mutation or a new boundary can
  invalidate them. Preserve useful compiler-visible information about valid
  types, ranges, alignment, and aliasing; assert only guarantees the contract
  actually provides.
- Use small templates, callable structs, local lambdas, and `if constexpr` to
  express genuine common structure and useful specialization. Introduce runtime
  polymorphism, registration, traits machinery, and policy layers for
  demonstrated requirements.
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

## Hot paths and compilation cost

**Prefer compile-time optimization and useful specialization over recurring
runtime overhead in hot paths; accept longer builds to obtain better execution.**

Spend compilation effort to remove recurring runtime work. Use available static
information and useful specialization through templates, static dispatch,
`if constexpr`, and compile-time knowledge of types, layouts, extents, and
alignment.
Keep implementations visible to the compiler where that enables optimization.
Compilation expense alone does not justify introducing runtime overhead into
hot paths. Compile-time machinery earns its place through
useful structure, specialization, or elimination of repeated code.

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
