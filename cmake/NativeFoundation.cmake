# Linked numerical foundation used by external native consumers.  This preserves
# the prepared-program/session and relation owners; it does not create a runner.
add_subdirectory(src/compute/operation/native_numeric)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/indexed_mechanism/CMakeLists.txt")
    add_subdirectory(src/compute/operation/indexed_mechanism)
endif()

add_library(cellerator_native_foundation INTERFACE)
add_library(Cellerator::native_foundation ALIAS cellerator_native_foundation)
target_link_libraries(cellerator_native_foundation INTERFACE
    Cellerator::native_numeric
    Cellerator::executable_program)

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
    if(TARGET Cellerator::indexed_mechanism)
        add_subdirectory(tests/native_foundation/nary)
    endif()
    if(TARGET Cellerator::native_value_instance)
        add_subdirectory(tests/native_foundation/values)
    endif()
endif()

# Export the actual linked dependency closure, including the preserved program
# and candidate owners. Consumers use these libraries, never source inclusion.
set(_nf_pending cellerator_native_foundation)
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
