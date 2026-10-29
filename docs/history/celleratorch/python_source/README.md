# Preserved CelleraTorch Python source

This directory preserves the former CelleraTorch Python package and its two
Python test files. It is a historical source snapshot, not the current
installed API, a supported package, or a new feature. The archived tests are
stored with `.txt` suffixes so ordinary pytest discovery does not collect them.
They have not been run as part of this preservation.

The three package modules were copied byte-for-byte from the surviving ignored
build output at `components/CelleraTorch/python/build/lib/celleratorch/`.
Their bytes match the original tracked files at Git commit
`84796f838fb32a11fbd37a147ad66c523285ef68`. At that commit the original
package paths were `components/CelleraTorch/python/celleratorch/`, the tests
were under `components/CelleraTorch/tests/`, and the package metadata was
`components/CelleraTorch/python/pyproject.toml`.

`docs/learning/bio-evidence/manifest.json` records the hashes for the three
modules and `test_biology.py`. All four match the snapshot below. Its
`source_commit` field is `009a5579c1facec8c3bce3c36f6764c9f49ab97c`, but that
tree does not contain `biology.py` or `test_biology.py` at those paths, and its
`__init__.py` hash differs. The byte-identical complete source and tests are
present in Git at `84796f838fb32a11fbd37a147ad66c523285ef68`; the earlier recorded tree's
`mechanism.py`, `test_mechanism.py`, and `pyproject.toml` also match that
commit. The manifest's per-file hashes identify the captured source; its HEAD
field alone does not imply that every captured input was already committed.

| Snapshot file | Former path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `celleratorch/__init__.py` | `components/CelleraTorch/python/celleratorch/__init__.py` | 844 | `3cf269bf8446e0a8d901dbbf88aad3497a91c837876061d52c6215b0aa1dd24a` |
| `celleratorch/biology.py` | `components/CelleraTorch/python/celleratorch/biology.py` | 7277 | `e4a77615284fae67d608f9f00ca9ebbd23efcb51ce6305b46f0d1543bb5a2cf4` |
| `celleratorch/mechanism.py` | `components/CelleraTorch/python/celleratorch/mechanism.py` | 36541 | `d627e92d771ee7a917f57aadf03cf0537663c7fde882677c2b0d06195e9e6d43` |
| `tests/test_biology.py.txt` | `components/CelleraTorch/tests/test_biology.py` | 11878 | `140281ba6bbc0e98c63dd45d005cd4eee1ec5ec311a38818b271d43649d99a51` |
| `tests/test_mechanism.py.txt` | `components/CelleraTorch/tests/test_mechanism.py` | 20950 | `3d66661cc0b66638b416985a8441ee0805b248040612dc99ef283e212f1809ea` |
| `pyproject.toml` | `components/CelleraTorch/python/pyproject.toml` | 372 | `84ed94fd8eae8c3b59778e6d5e3d654c11c379df85d14490260292601996f2ff` |

The module and test files passed Python AST parsing and the package metadata
parsed as TOML. These checks establish syntactic readability only; they do not
qualify the archived package or its behavior.
