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
