# Linked numerical foundation used by external native consumers.  This preserves
# the prepared-program/session and relation owners; it does not create a runner.
add_subdirectory(src/compute/operation/native_numeric)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/differential/CMakeLists.txt")
    add_subdirectory(src/compute/operation/differential)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/indexed_mechanism/CMakeLists.txt")
    add_subdirectory(src/compute/operation/indexed_mechanism)
endif()

add_library(cellerator_native_foundation INTERFACE)
add_library(Cellerator::native_foundation ALIAS cellerator_native_foundation)
target_link_libraries(cellerator_native_foundation INTERFACE
    Cellerator::native_numeric
    Cellerator::executable_program)
if(TARGET Cellerator::model_ops)
    target_link_libraries(cellerator_native_foundation INTERFACE
        Cellerator::model_ops)
endif()
if(TARGET Cellerator::local_differential)
    target_link_libraries(cellerator_native_foundation INTERFACE Cellerator::local_differential)
endif()

if(TARGET Cellerator::indexed_mechanism)
    target_link_libraries(cellerator_native_foundation INTERFACE
        Cellerator::indexed_mechanism)
endif()

if(TARGET cellerator_prepared_relation_cuda)
    target_link_libraries(cellerator_native_foundation INTERFACE
        Cellerator::prepared_relation_cuda)
    # The historical V07 facade also requires atom-plane owners.  Preserve the
    # real CUDA pair whenever available; publish that facade only when its
    # existing owners are present rather than introducing a duplicate owner.
    if(TARGET Cellerator::jbc_v1 AND TARGET cellerator_compute_sparse_project)
        add_subdirectory(src/execution/native_value_instance)
        target_link_libraries(cellerator_native_foundation INTERFACE
            Cellerator::native_value_instance)
    endif()
endif()

if(CELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS)
    enable_testing()
    add_subdirectory(tests/native_foundation/reference)
    add_subdirectory(tests/native_foundation/numeric)
    add_subdirectory(tests/native_foundation/program)
    add_subdirectory(tests/native_foundation/build)
    if(TARGET Cellerator::local_differential)
        add_subdirectory(tests/native_foundation/differential)
    endif()
    if(TARGET Cellerator::indexed_mechanism)
        add_subdirectory(tests/native_foundation/nary)
    endif()
    if(TARGET Cellerator::relation_algebra AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/support/CMakeLists.txt")
        add_subdirectory(tests/native_foundation/support)
    endif()
    if(TARGET Cellerator::native_value_instance)
        add_subdirectory(tests/native_foundation/values)
    endif()
    if(TARGET Cellerator::local_differential)
        add_subdirectory(tests/native_foundation/integration)
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/native_foundation/CMakeLists.txt")
    add_subdirectory(bench/native_foundation)
endif()

# Export the actual linked dependency closure, including the preserved program
# and candidate owners. Consumers use these libraries, never source inclusion.
set(_nf_pending cellerator_native_foundation)
if(TARGET cellerator_training_program)
    # Keep the native training-program API available to optional bindings
    # without making the native owner depend on any framework adapter.
    list(APPEND _nf_pending cellerator_training_program)
endif()
set(_nf_exports)
while(_nf_pending)
    list(POP_FRONT _nf_pending _nf_target)
    if(NOT TARGET "${_nf_target}")
        continue()
    endif()
    get_target_property(_nf_alias "${_nf_target}" ALIASED_TARGET)
    if(_nf_alias)
        set(_nf_target "${_nf_alias}")
    endif()
    get_target_property(_nf_imported "${_nf_target}" IMPORTED)
    if(_nf_imported OR _nf_target IN_LIST _nf_exports)
        continue()
    endif()
    list(APPEND _nf_exports "${_nf_target}")
    foreach(_nf_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(_nf_links "${_nf_target}" ${_nf_property})
        foreach(_nf_link IN LISTS _nf_links)
            string(REGEX REPLACE "^\\$<LINK_ONLY:([^>]+)>$" "\\1" _nf_link "${_nf_link}")
            if(TARGET "${_nf_link}")
                list(APPEND _nf_pending "${_nf_link}")
            endif()
        endforeach()
    endforeach()
    if(_nf_target MATCHES "^cellerator_(.+)$")
        set_target_properties("${_nf_target}" PROPERTIES EXPORT_NAME "${CMAKE_MATCH_1}")
    endif()
    get_target_property(_nf_includes "${_nf_target}" INTERFACE_INCLUDE_DIRECTORIES)
    if(_nf_includes)
        set(_nf_public_includes)
        foreach(_nf_include IN LISTS _nf_includes)
            if(_nf_include STREQUAL "${PROJECT_SOURCE_DIR}/include")
                list(APPEND _nf_public_includes "$<BUILD_INTERFACE:${_nf_include}>" "$<INSTALL_INTERFACE:include>")
            elseif(_nf_include STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/generated")
                list(APPEND _nf_public_includes
                    "$<BUILD_INTERFACE:${_nf_include}>"
                    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
            else()
                list(APPEND _nf_public_includes "${_nf_include}")
            endif()
        endforeach()
        set_target_properties("${_nf_target}" PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_nf_public_includes}")
    endif()
endwhile()
install(TARGETS ${_nf_exports} EXPORT CelleratorNativeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorNativeTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
export(EXPORT CelleratorNativeTargets NAMESPACE Cellerator::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/CelleratorNativeTargets.cmake")
# Extend the host package rather than replace its existing compiler contract.
file(APPEND "${CELLERATOR_HOST_CONFIG}"
    "include(CMakeFindDependencyMacro)\nfind_dependency(CUDAToolkit)\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorNativeTargets.cmake\")\n"
    "set(Cellerator_native_foundation_FOUND TRUE)\nset(Cellerator_HAS_CUDA_BACKEND TRUE)\n")
export(EXPORT CelleratorPartOneTargets NAMESPACE Cellerator::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/CelleratorPartOneTargets.cmake")

# Bind downstream replay provenance to the installed producer build, rather
# than to a caller-provided runtime string. Reconfigure after source commits.
find_package(Git QUIET)
set(Cellerator_BUILD_SOURCE_REVISION "unversioned")
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE Cellerator_BUILD_SOURCE_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nf_revision_status)
    if(NOT _nf_revision_status EQUAL 0)
        set(Cellerator_BUILD_SOURCE_REVISION "unversioned")
    endif()
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CelleratorBuildIdentity.cmake"
    "set(Cellerator_BUILD_SOURCE_REVISION \"${Cellerator_BUILD_SOURCE_REVISION}\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CelleratorBuildIdentity.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
file(APPEND "${CELLERATOR_HOST_CONFIG}"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorBuildIdentity.cmake\")\n")

# Retain the existing independent contracts in the NF1A registered inventory.
if(CELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS AND CELLERATOR_BUILD_TESTS)
    set(_nf_retained_pairs
        "ss1_core|ceSpineCoreTest"
        "ss1_duality|ceSpineMathematicalDualityTest"
        "ss1_algebra|ceSpineAlgebraTest"
        "ss1_contract|ceSpineContractProbes"
        "program|celleratorExecutableProgramTest"
        "gate|ceGeoEdgeMapOrGateTest"
        "segment_reduce|ceGeoSegmentReduceTest"
        "segment_normalize|ceGeoSegmentNormalizeTest")
    foreach(_nf_pair IN LISTS _nf_retained_pairs)
        string(REPLACE "|" ";" _nf_parts "${_nf_pair}")
        list(GET _nf_parts 0 _nf_name)
        list(GET _nf_parts 1 _nf_binary)
        if(NOT TARGET ${_nf_binary})
            message(FATAL_ERROR "Required NF1A retained target missing: ${_nf_binary}")
        endif()
        add_test(NAME ce_nf1_retained_${_nf_name} COMMAND $<TARGET_FILE:${_nf_binary}>)
        set_tests_properties(ce_nf1_retained_${_nf_name} PROPERTIES TIMEOUT 180)
    endforeach()
endif()
