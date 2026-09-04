# Standard-library package structure v1

The installed `.cell` standard library is rooted at `share/cellerator/stdlib`
and uses these acyclic layers:

1. `core`: semantic wrappers and fundamental views;
2. `biology`: common domain and relation declarations;
3. `operations`: reusable algorithms over typed relations and state;
4. `planning`: profiles, planning helpers, and reflection utilities;
5. `interop`: explicit native and backend-facing adapters.

Each package exposes one same-named umbrella source and may import only an
earlier layer. Applications import installed package names; they never depend
on build-tree paths. Compiler-owned syntax and CEIR definitions are not copied
into the standard library.
