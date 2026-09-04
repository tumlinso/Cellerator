# Language and IR authority document inventory

Todo: `CE-CCP1-A01-003`

The inventory covers every regular file under `docs/language` in the claimed
A01 workspace. SHA-256 values were computed from exact file bytes. Repository
status was checked with `git status --short --untracked-files=all --
docs/language` and `git ls-files --stage -- docs/language`.

## Inventory

| Path | Bytes | SHA-256 | Git status | Git blob |
|---|---:|---|---|---|
| `docs/language/cellerator-language-specification.md` | 104647 | `22329eef2e84e6b48d7d304e77c2ed65b75f29721d65f499680faa08c8efe15b` | tracked, index/worktree clean | `2efb95ced37fd3f2a7a37b75b06102c1efb7acaf` |
| `docs/language/cellerator-programming-guide.md` | 77857 | `5fd78c016fddf3876e20c291affd9042f9dfc854ac520c0834dcb20663e20c26` | tracked, index/worktree clean | `99b9b475d3ee46d63b622f1e73248fa49c64095a` |

Named-source disposition:

- Cellerator language specification: present and hashed.
- Cellerator programming guide: present and hashed.
- Externally supplied standalone IR design document: absent from
  `docs/language`; no external document was indexed because no exact bytes were
  available in the authoritative read scope. IR design present inside the two
  listed documents remains covered by their whole-file hashes.

## Heading outline: language specification

Heading counts: H1 1, H2 30, H3 165, H4-H6 0.

- Cellerator Programming Language: Proposed Specification
  - Contents
  - 1. Purpose and scope
  - 2. Normative vocabulary
  - 3. Design principles
  - 4. Relationship to C++
  - 5. Lexical and grammatical additions
  - 6. Compiler-semantic types
  - 7. Biological typing and explicit escape hatches
  - 8. Relation transfer expressions
  - 9. Operation families
  - 10. Execution fields
  - 11. Planning directives and authority hierarchy
  - 12. Representative data and data-state evolution
  - 13. Persistence, reuse, identity, and biological time
  - 14. Numerical, determinism, output, and order contracts
  - 15. C, C++, CUDA, and native interoperability
  - 16. Exact coverage, decomposition, atoms, and extents
  - 17. Custom candidates, cost models, and forced realization
  - 18. Intermediate representation as a programming feature
  - 19. Diagnostics and introspection
  - 20. Errors, warnings, and fallback
  - 21. Compilation model
  - 22. Standard-library boundary
  - 23. Compact grammar sketch
  - 24. Integrated examples
  - 25. Implementation-defined behavior
  - 26. Extension and versioning strategy
  - 27. Rejected or avoided designs
  - 28. Open Design Questions
  - 29. Research grounding

## Heading outline: programming guide

Heading counts: H1 1, H2 36, H3 129, H4-H6 0.

- Programming Cellerator
  - Contents
  - 1. The programming model in one page
  - 2. Enabling Cellerator
  - 3. Domains, axes, state, and relations
  - 4. Your first relation computation
  - 5. What happens inside the compiler
  - 6. Execution fields
  - 7. Data-aware compilation
  - 8. Branches and bounded profile alternatives
  - 9. Persistence and reuse
  - 10. Hints, preferences, requirements, offers, and force
  - 11. Working with ordinary C++ and CUDA
  - 12. Understanding opaque-barrier warnings
  - 13. Numerical and deterministic programming
  - 14. Order, packing, and canonical boundaries
  - 15. Inspecting what Cellerator did
  - 16. IR inspection
  - 17. Writing an IR transform
  - 18. Explicit decomposition
  - 19. Atoms and extents
  - 20. Providing a custom candidate
  - 21. Competing with Cellerator, then forcing it
  - 22. Custom cost and external planning
  - 23. Asynchronous execution, readiness, and publication
  - 24. Forward, transpose, gradients, and training-shaped programs
  - 25. The same computation at five control levels
  - 26. A complete worked example
  - 27. Common mistakes
  - 28. Operation-family cookbook
  - 29. A productive optimization workflow
  - 30. Building Cellerator libraries
  - 31. Compilation and artifact workflow
  - 32. Returning to ordinary C++
  - 33. Performance philosophy in practice
  - 34. Reading the two documents together
  - 35. Source grounding

## Validation receipt

`find docs/language -type f` returned exactly the two paths above. A second
`sha256sum` pass reproduced both recorded hashes, and file sizes were read from
the filesystem after hashing. The heading counts and H1/H2 outline were derived
from lines matching Markdown ATX headings. No language document had an
untracked, staged, or modified status.
