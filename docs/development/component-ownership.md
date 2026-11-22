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

The former [AnnData++ and Velocity source archives](https://github.com/tumlinso/Cellerator/tree/d7414e7f830fcd25cac585399a0535b72e75ff01/quarantine)
remain available in Git history. Local cutover recovery files and the retired
Python AnnData source snapshot with its checksum manifest were backed up
outside the checkout.

The former `src/velocity` checkout was archived outside the repository with
its unpublished changes, Git metadata, and validation artifacts. CCell owns
direct binary16 input and caller-scheduled gene blocks.
