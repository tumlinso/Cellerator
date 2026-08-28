# C++20 module infrastructure probe

The 2026-08-28 probe established two separate facts:

- Clang 18 successfully precompiled `module_probe.ccm`, compiled its BMI to an
  object, compiled a host consumer with the prebuilt module path, linked the
  objects, and ran the consumer.
- CMake 3.28 rejected native `FILE_SET CXX_MODULES` generation with the
  repository's Unix Makefiles generator. CMake reports that native module
  scanning requires Ninja, Ninja Multi-Config, or a supported Visual Studio
  generator. Ninja and `clang-scan-deps` are not installed on this host.

The optional `celleratorModuleCompilerProbe` target preserves the successful
direct compiler proof without placing any CUDA target on a BMI dependency
graph. It is not represented as a successful native CMake scan.

No architectural module surface is enabled until a native CMake scan and host
consumer pass with installed repository tooling. Physical consolidation
continues through conventional `.hh`, `.cc`, and `.cu` contracts as required by
the remap program.
