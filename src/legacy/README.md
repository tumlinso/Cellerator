# `legacy`

Quarantine area for monolithic or transitional source files.

Current policy:

- keep old code alive until replacements exist
- extract reusable compute outward into `src/compute/*`
- do not route new reusable code through legacy entrypoints
