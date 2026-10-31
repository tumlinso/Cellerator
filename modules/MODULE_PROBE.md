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

## Native NN module, 2026-10-31

The original constant-export probe remains a historical infrastructure check.
The optional host [native NN module](native/README.md) adds a separate real
modeling surface. Its qualification uses Clang 18, CMake 3.28, and local Ninja
and clang-scan-deps tools, native dependency scanning, installed consumers,
ATen/autograd, and a typed model. See the source-bound
[module record](../planning/native-nn-foundation-2026-10-31/modules/qualification.md)
for the actual supported configurations and toolchain limits.

This does not qualify the broader deferred state/execution/interop modules.
Conventional `.hh`, `.cc`, and `.cu` contracts remain independently usable;
CUDA translation units do not import named modules.
