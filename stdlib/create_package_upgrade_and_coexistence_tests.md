# Package upgrade and coexistence v1

Resources install below `share/cellerator/<compiler-major>.<ceir-major>`.
Compiler, resource tree, profile, IR, and plugin declarations carry both major
identities; loading requires an exact pair. Diagnostics name both requested and
found versions. Search never falls through to a different versioned tree.
