# Source-linked existing owners; no independent execution implementation.
include_guard(GLOBAL)
get_filename_component(_nf1_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(cellerator_operation_schema_v2 STATIC
    "${_nf1_root}/src/compute/operation/operation_core_v2/schema.cc")
add_library(Cellerator::operation_schema_v2 ALIAS cellerator_operation_schema_v2)
add_library(cellerator_prepared_program_v2 STATIC
    "${_nf1_root}/src/execution/program/program_v2.cc")
add_library(Cellerator::prepared_program_v2 ALIAS cellerator_prepared_program_v2)
foreach(_nf1_target cellerator_operation_schema_v2 cellerator_prepared_program_v2)
    target_include_directories(${_nf1_target} PUBLIC "${_nf1_root}/include")
    target_compile_features(${_nf1_target} PUBLIC cxx_std_17)
    set_target_properties(${_nf1_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()
# The C++20 contract surface composes the two compiled C++17 owners.
add_library(cellerator_native_foundation INTERFACE)
add_library(Cellerator::native_foundation ALIAS cellerator_native_foundation)
target_link_libraries(cellerator_native_foundation INTERFACE
    Cellerator::operation_schema_v2 Cellerator::prepared_program_v2)
target_compile_features(cellerator_native_foundation INTERFACE cxx_std_20)

# Rehome existing host owners without copying their implementation into consumers.
add_library(cellerator_relation_semantics STATIC "${_nf1_root}/src/compute/operation/relation_semantics.cc")
add_library(Cellerator::relation_semantics ALIAS cellerator_relation_semantics)
add_library(cellerator_relation_calculus STATIC "${_nf1_root}/src/compute/operation/relation_calculus.cc")
add_library(Cellerator::relation_calculus ALIAS cellerator_relation_calculus)
target_link_libraries(cellerator_relation_calculus PUBLIC Cellerator::relation_semantics)
add_library(cellerator_segment_host STATIC
    "${_nf1_root}/src/compute/candidate/segment/segment_v2.cc"
    "${_nf1_root}/src/compute/candidate/segment/reduce_v2_reference.cc"
    "${_nf1_root}/src/compute/candidate/segment/normalize_v2_reference.cc")
add_library(Cellerator::segment_host ALIAS cellerator_segment_host)
add_library(cellerator_gate_validation STATIC
    "${_nf1_root}/src/compute/candidate/edge/gate_update_validation_v1.cc")
add_library(Cellerator::gate_validation ALIAS cellerator_gate_validation)
foreach(_nf1_target cellerator_relation_semantics cellerator_relation_calculus cellerator_segment_host cellerator_gate_validation)
    target_include_directories(${_nf1_target} PUBLIC "${_nf1_root}/include")
    target_compile_features(${_nf1_target} PUBLIC cxx_std_17)
    set_target_properties(${_nf1_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()
target_link_libraries(cellerator_native_foundation INTERFACE
    Cellerator::relation_calculus Cellerator::segment_host Cellerator::gate_validation)

# Call after declaring consumer sources. Private implementation inclusions bypass
# target ownership and produce duplicate or unqualified implementations.
function(cellerator_link_native_foundation consumer)
    get_target_property(_sources ${consumer} SOURCES)
    get_target_property(_source_dir ${consumer} SOURCE_DIR)
    foreach(_source IN LISTS _sources)
        if(NOT IS_ABSOLUTE "${_source}")
            set(_source "${_source_dir}/${_source}")
        endif()
        if(EXISTS "${_source}")
            file(READ "${_source}" _text)
            if(_text MATCHES "#[ \t]*include[ \t]*[<\"][^>\"]+\\.(cu|cc|cpp)[>\"]")
                message(FATAL_ERROR "Native consumer ${consumer} includes a private implementation: ${_source}; link declared Cellerator targets instead")
            endif()
        endif()
    endforeach()
    target_link_libraries(${consumer} PRIVATE Cellerator::native_foundation)
endfunction()

# Device execution stays in the established prepared-relation, segment/gate,
# and value-readiness owners. No host replacement satisfies this dependency.
function(cellerator_link_native_cuda consumer)
    foreach(_owner Cellerator::prepared_relation_cuda Cellerator::relation_algebra Cellerator::runtime)
        if(NOT TARGET ${_owner})
            message(FATAL_ERROR "${_owner} unavailable: configure CUDA and CELLERATOR_BUILD_SEMANTIC_SPINE_V1=ON")
        endif()
    endforeach()
    cellerator_link_native_foundation(${consumer})
    target_link_libraries(${consumer} PRIVATE Cellerator::prepared_relation_cuda
        Cellerator::relation_algebra Cellerator::runtime)
endfunction()

# N lane owns this implementation. AUTO admits it once integrated; explicit ON
# fails if its source fragment is unavailable, rather than qualifying an empty test.
set(CELLERATOR_BUILD_NATIVE_NUMERIC "AUTO" CACHE STRING "Build native numeric targets: AUTO, ON, OFF")
set_property(CACHE CELLERATOR_BUILD_NATIVE_NUMERIC PROPERTY STRINGS AUTO ON OFF)
if(NOT CELLERATOR_BUILD_NATIVE_NUMERIC MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "CELLERATOR_BUILD_NATIVE_NUMERIC must be AUTO, ON or OFF")
endif()
set(_nf1_numeric FALSE)
if(CELLERATOR_BUILD_NATIVE_NUMERIC STREQUAL "ON" OR
   (CELLERATOR_BUILD_NATIVE_NUMERIC STREQUAL "AUTO" AND
    EXISTS "${_nf1_root}/src/compute/operation/native_numeric/CMakeLists.txt"))
    add_subdirectory("${_nf1_root}/src/compute/operation/native_numeric" "${CMAKE_CURRENT_BINARY_DIR}/nf1-numeric")
    set(_nf1_numeric TRUE)
endif()
if(CELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS)
    enable_testing()
    add_subdirectory("${_nf1_root}/tests/native_foundation/reference" "${CMAKE_CURRENT_BINARY_DIR}/nf1-reference-tests")
    add_subdirectory("${_nf1_root}/tests/native_foundation/build" "${CMAKE_CURRENT_BINARY_DIR}/nf1-build-tests")
    if(_nf1_numeric)
        add_subdirectory("${_nf1_root}/tests/native_foundation/numeric" "${CMAKE_CURRENT_BINARY_DIR}/nf1-numeric-tests")
    endif()
endif()
