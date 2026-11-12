include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Native package surface. Keep language SDK headers and resources out of both
# the regular and selectable substrate installations.
function(cellerator_install_native_headers)
    foreach(dir IN ITEMS bindings biology cells common compute containers
            examples execution geometry interop layout math memory modeling nn
            ops packing planner profiling runtime state structure tensor)
        install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/Cellerator/${dir}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN})
    endforeach()
    install(FILES
        "${PROJECT_SOURCE_DIR}/include/Cellerator/Cellerator.hh"
        "${PROJECT_SOURCE_DIR}/include/Cellerator/abi.h"
        "${PROJECT_SOURCE_DIR}/include/Cellerator/cell.hh"
        "${PROJECT_SOURCE_DIR}/include/Cellerator/nn.hh"
        "${PROJECT_SOURCE_DIR}/include/Cellerator/parameters.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN})
endfunction()

add_library(libCellerator STATIC src/cell.cc)
add_library(Cellerator::libCellerator ALIAS libCellerator)
target_include_directories(libCellerator PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(libCellerator PUBLIC cxx_std_20)
set_target_properties(libCellerator PROPERTIES OUTPUT_NAME Cellerator)
add_library(cellerator_cpu_numerics STATIC
    src/compute/cpu/generic_cpu_relation_apply.cc
    src/compute/cpu/cpu_transpose_and_contraction.cc
    src/compute/cpu/cpu_projection_packing_and_order_transforms.cc
    src/compute/cpu/cpu_segment_gate_update_bundle_and_chain_paths.cc)
add_library(Cellerator::cpu_numerics ALIAS cellerator_cpu_numerics)
target_include_directories(cellerator_cpu_numerics PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_cpu_numerics PUBLIC cxx_std_20)
set_target_properties(cellerator_cpu_numerics PROPERTIES EXPORT_NAME cpu_numerics)
install(TARGETS cellerator_cpu_numerics EXPORT CelleratorCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS libCellerator EXPORT CelleratorCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorCoreTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
cellerator_install_native_headers()

set(CELLERATOR_HOST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfig.cmake")
file(WRITE "${CELLERATOR_HOST_CONFIG}"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorCoreTargets.cmake\")\n"
    "set(Cellerator_HAS_RUNTIME TRUE)\nset(Cellerator_HAS_CUDA_BACKEND FALSE)\n")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfigVersion.cmake"
    VERSION 1.0.0 COMPATIBILITY SameMajorVersion)
install(FILES "${CELLERATOR_HOST_CONFIG}"
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cell)
    foreach(name IN ITEMS generic_cpu_relation_apply cpu_transpose_and_contraction
            cpu_projection_packing_and_order_transforms
            cpu_segment_gate_update_bundle_and_chain_paths)
        add_executable(cellerator_${name}_test tests/compute/cpu/${name}_test.cc)
        target_link_libraries(cellerator_${name}_test PRIVATE Cellerator::cpu_numerics)
        add_test(NAME cellerator_${name}_test COMMAND cellerator_${name}_test)
    endforeach()
    foreach(example event_forward multimodal)
        set(target "cellerator_cell_${example}_example")
        add_executable(${target} "${PROJECT_SOURCE_DIR}/examples/native_cell/${example}.cc")
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_link_libraries(${target} PRIVATE Cellerator::libCellerator)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "native_cell")
    endforeach()
    add_executable(cellerator_cell_foundation_example
        "${PROJECT_SOURCE_DIR}/examples/native_cell/foundation.cc")
    target_compile_features(cellerator_cell_foundation_example PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(cellerator_cell_foundation_example PRIVATE /GR-)
    else()
        target_compile_options(cellerator_cell_foundation_example PRIVATE -fno-rtti)
    endif()
    add_test(NAME cellerator_cell_foundation_example COMMAND cellerator_cell_foundation_example)
    set_tests_properties(cellerator_cell_foundation_example PROPERTIES LABELS "native_cell")
endif()
