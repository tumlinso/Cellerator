# Clean host-only SDK acceptance

Configured with `CELLERATOR_ENABLE_CUDA=OFF`, built and installed `cellerator`,
`libCellerator`, `celleratord`, public headers, standard-library resources,
reference profiles, and relocatable CMake metadata. The host-only path returns
before legacy CUDA runtime targets are declared and does not load the CUDA
language. Compiler smoke and installed consumer checks use only C++17.
