# Working on CCell

CCell owns single-cell preprocessing and RNA velocity. Follow the root
[protected C++ coding style](../../coding_style.md). Use the [CCell build and API
guide](README.md) for source routes, installed targets, and import examples.

- Put reusable layouts, views, and mathematical primitives with their existing
  Cellerator owners. Keep scientific method behavior in `src/preprocess/` or
  `src/velocity/`. Keep data acquisition and translation with the optional
  AnnData++ adapter in `src/ccell.interop.anndata.cppm`.
- Numerical code accepts Cellerator's typed dense and compressed views. Dense
  strides count elements and can be signed. Sparse views carry CSR or CSC
  orientation and independently typed indices and offsets. Treat observation
  and feature ordering as part of each input contract.
- Use `import anndata;` for AnnData++ I/O and `import ccell.interop.anndata;` for
  checked borrowing. Within an adapter implementation, use
  `namespace anndata = ::anndatapp;` locally when it makes code clearer. Build
  and package names remain `anndatapp` and `anndatapp::anndatapp`.
- Borrowed views do not extend storage lifetimes. Retain owners through all
  numerical reads, including reads from prepared Velocity inputs. Callers
  provide output, preparation storage, scratch, and synchronization. Keep any
  copying, conversion, materialization, and allocation explicit. Numerical
  methods reuse sufficient storage without allocation.
- Keep public module interfaces focused and compiled implementation details
  private. Test native method code with AnnData++ disabled. Test changes to the
  adapter using the standard build and an installed consumer.

Keep the original source and qualification records attributable. The imported
[CCell style record](coding_style.md) and Velocity's historical evidence retain
their original context. Follow their recorded precision and scientific
contracts, and use local tests to verify the moved source.
