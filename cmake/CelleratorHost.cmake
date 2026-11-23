include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Native package surface. Keep language SDK headers and resources out of both
# the regular and selectable substrate installations.
function(cellerator_install_native_headers)
    foreach(dir IN ITEMS bindings biology common compute containers
            execution geometry interop layout math memory nn
            ops packing planner profiling runtime state structure tensor)
        install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/Cellerator/${dir}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN}
            PATTERN "cellshard" EXCLUDE)
    endforeach()
    install(FILES
        "${PROJECT_SOURCE_DIR}/include/Cellerator/nn.hh"
        "${PROJECT_SOURCE_DIR}/include/Cellerator/parameters.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN})
endfunction()

add_library(cellerator_headers INTERFACE)
add_library(Cellerator::headers ALIAS cellerator_headers)
set_target_properties(cellerator_headers PROPERTIES EXPORT_NAME headers)
target_include_directories(cellerator_headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_headers INTERFACE cxx_std_20)
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
install(TARGETS cellerator_headers EXPORT CelleratorCoreTargets)
install(EXPORT CelleratorCoreTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
cellerator_install_native_headers()
include("${PROJECT_SOURCE_DIR}/examples/install_headers.cmake")

set(CELLERATOR_HOST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/CelleratorConfig.cmake")
file(WRITE "${CELLERATOR_HOST_CONFIG}"
    "foreach(component IN LISTS Cellerator_FIND_COMPONENTS)\n"
    "  if(component MATCHES \"^(velocity|cellshard|CellShard)$\")\n"
    "    set(Cellerator_FOUND FALSE)\n"
    "    set(Cellerator_NOT_FOUND_MESSAGE \"The requested \${component} component is retired from Cellerator; retained data/analysis source is inactive.\")\n"
    "    return()\n"
    "  endif()\n"
    "endforeach()\n"
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
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME cellerator_core_stl_policy_test
            COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/style/test_core_stl_policy.py")
    endif()
endif()
foreach(name IN ITEMS generic_cpu_relation_apply cpu_transpose_and_contraction
        cpu_projection_packing_and_order_transforms
        cpu_segment_gate_update_bundle_and_chain_paths)
    add_executable(cellerator_${name}_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        tests/compute/cpu/${name}_test.cc)
    target_link_libraries(cellerator_${name}_test PRIVATE Cellerator::cpu_numerics)
    if(CELLERATOR_BUILD_TESTS)
        add_test(NAME cellerator_${name}_test COMMAND cellerator_${name}_test)
    endif()
endforeach()
foreach(example event_forward multimodal)
    set(target "cellerator_cell_${example}_example")
    add_executable(${target} ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
        "${PROJECT_SOURCE_DIR}/examples/native_cell/${example}.cc")
    target_link_libraries(${target} PRIVATE Cellerator::headers)
    if(CELLERATOR_BUILD_TESTS)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "native_cell")
    endif()
endforeach()
