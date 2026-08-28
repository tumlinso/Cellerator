# C++20 modules are source-level semantic borders, not a dependency of the
# native CUDA graph. The installed Clang compiler can precompile and import the
# probe, but native CMake scanning is deferred until a supported generator and
# clang-scan-deps are available. This direct compiler probe remains optional and
# leaves the repository's established GNU/NVCC compiler pairing unchanged.

find_program(CELLERATOR_MODULE_CLANGXX NAMES clang++-18)
if(CELLERATOR_MODULE_CLANGXX)
    set(cellerator_module_probe_build "${CMAKE_CURRENT_BINARY_DIR}/probe-build")
    set(cellerator_module_probe_pcm
        "${cellerator_module_probe_build}/cellerator.module_probe.pcm")

    add_custom_target(celleratorModuleCompilerProbe
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${cellerator_module_probe_build}"
        COMMAND "${CELLERATOR_MODULE_CLANGXX}" -std=c++20 --precompile
            "${CMAKE_CURRENT_SOURCE_DIR}/probe/module_probe.ccm"
            -o "${cellerator_module_probe_pcm}"
        COMMAND "${CELLERATOR_MODULE_CLANGXX}" -std=c++20 -c
            "${cellerator_module_probe_pcm}"
            -o "${cellerator_module_probe_build}/module_probe.o"
        COMMAND "${CELLERATOR_MODULE_CLANGXX}" -std=c++20
            "-fprebuilt-module-path=${cellerator_module_probe_build}" -c
            "${CMAKE_CURRENT_SOURCE_DIR}/probe/module_probe_test.cc"
            -o "${cellerator_module_probe_build}/module_probe_test.o"
        COMMAND "${CELLERATOR_MODULE_CLANGXX}"
            "${cellerator_module_probe_build}/module_probe_test.o"
            "${cellerator_module_probe_build}/module_probe.o"
            -o "${cellerator_module_probe_build}/module_probe_test"
        COMMAND "${cellerator_module_probe_build}/module_probe_test"
        COMMENT "Precompiling and importing the optional host-only module probe"
        USES_TERMINAL
        VERBATIM)
else()
    add_custom_target(celleratorModuleCompilerProbe
        COMMAND "${CMAKE_COMMAND}" -E echo
            "Clang 18 is required for the optional host-only module probe"
        COMMAND "${CMAKE_COMMAND}" -E false
        VERBATIM)
endif()
