# CCell and Cellerator component ownership

CCell owns the native preprocessing and RNA velocity implementations under
[`components/CCell`](../../components/CCell/README.md). Cellerator supplies
their numerical and modeling primitives and view contracts. AnnData++ owns
arrays and file I/O. CCell's optional adapter binds acquired AnnData++ arrays
to Cellerator views for use by either method.

The standard build installs CCell and AnnData++ together. `find_package(ccell
CONFIG REQUIRED)` loads the method targets. Request the `anndata_interop`
component for CCell's borrowing adapter. AnnData++ itself supplies
`import anndata;`. The Cellerator core package can be discovered independently.
