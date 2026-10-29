# Compiler resource discovery v1

The compiler canonicalizes its executable path, takes the parent installation
prefix, and locates versioned `share/cellerator` resources and sibling support
binaries from that prefix. Dedicated `--stdlib`, `--profiles`, `--backends`,
`--schemas`, and `--support-bin` flags override only their named resource.
Neither the current directory nor source/build-tree paths participate.
