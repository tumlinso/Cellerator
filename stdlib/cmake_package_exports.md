# CMake package exports v1

The relocatable package exposes `Cellerator::Compiler`, `Runtime`,
`BackendCUDA`, and `ProviderSDK`, plus compiler/runtime/backend feature and
CEIR/profile version variables. Dependencies resolve through installed package
metadata. Exported properties contain no source- or build-tree paths.
