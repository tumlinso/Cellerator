# Installable SDK and standard-library foundation v1

The installed foundation supports ordinary C++, `.cell`, standalone CEIR,
direct libCellerator, custom-pass, CPU, and conditional NVCC consumers. The SDK
package and stdlib are version-paired, executable/prefix relative, and require
an explicit profile. `stdlib/manifest.json` hashes installed sources;
`stdlib/cellerator/core.cell` is the stable umbrella. The compiler/runtime APIs
remain those frozen by I35/I36, and profile artifacts remain I17-compatible.
