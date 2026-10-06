cmake_minimum_required(VERSION 3.21)
project(cellerator_fp64_qualification LANGUAGES CXX)

if(NOT CELLERATOR_SOURCE_DIR)
    get_filename_component(CELLERATOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()
set(CELLERATOR_ENABLE_CUDA ON CACHE STRING "" FORCE)
set(CELLERATOR_BUILD_SEMANTIC_SPINE_V1 ON CACHE BOOL "" FORCE)
set(CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1 OFF CACHE BOOL "" FORCE)
set(CELLERATOR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CELLERATOR_CUDA_ARCHITECTURES 70 CACHE STRING "" FORCE)
add_subdirectory("${CELLERATOR_SOURCE_DIR}" cellerator EXCLUDE_FROM_ALL)
enable_language(CUDA)

add_executable(fp64_qualification fp64_qualification.cu)
target_include_directories(fp64_qualification PRIVATE "${CELLERATOR_SOURCE_DIR}/include")
target_compile_features(fp64_qualification PRIVATE cxx_std_17)
target_compile_options(fp64_qualification PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG> $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
set_target_properties(fp64_qualification PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)
target_link_libraries(fp64_qualification PRIVATE
    Cellerator::prepared_relation_cuda cellerator_compute_sparse_project)

function(add_legacy_native_witness name)
    file(READ "${CELLERATOR_SOURCE_DIR}/tests/semantic_spine/native/${name}_test.cu" witness_source)
    string(REPLACE "../../../src/compute/operation/prepared_relation.cu"
        "${CELLERATOR_SOURCE_DIR}/src/compute/operation/prepared_relation.cu"
        witness_source "${witness_source}")
    string(REPLACE "p->logical_map" "p->structure->logical_map"
        witness_source "${witness_source}")
    set(witness_copy "${CMAKE_CURRENT_BINARY_DIR}/compat_${name}_test.cu")
    file(WRITE "${witness_copy}" "${witness_source}")
    add_executable(fp64_existing_${name}_test "${witness_copy}")
    target_include_directories(fp64_existing_${name}_test PRIVATE
        "${CELLERATOR_SOURCE_DIR}/include"
        "${CELLERATOR_SOURCE_DIR}/tests/semantic_spine/native")
    target_compile_features(fp64_existing_${name}_test PRIVATE cxx_std_17)
    target_compile_options(fp64_existing_${name}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG> $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
    set_target_properties(fp64_existing_${name}_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES 70)
    target_link_libraries(fp64_existing_${name}_test PRIVATE
        cellerator_relation_update_support Cellerator::prepared_relation_cuda)
endfunction()
foreach(name adapter forward transpose lifecycle overhead)
    add_legacy_native_witness(${name})
endforeach()

add_executable(fp64_admission_test admission_test.cu)
target_include_directories(fp64_admission_test PRIVATE
    "${CELLERATOR_SOURCE_DIR}/include"
    "${CELLERATOR_SOURCE_DIR}/tests/semantic_spine/native")
target_compile_features(fp64_admission_test PRIVATE cxx_std_17)
target_compile_options(fp64_admission_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG> $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
set_target_properties(fp64_admission_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)
target_link_libraries(fp64_admission_test PRIVATE
    cellerator_relation_update_support Cellerator::prepared_relation_cuda)

add_executable(fp64_cost_bench cost_bench.cu)
target_include_directories(fp64_cost_bench PRIVATE
    "${CELLERATOR_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/cellerator/generated")
target_compile_features(fp64_cost_bench PRIVATE cxx_std_17)
set_target_properties(fp64_cost_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)
target_link_libraries(fp64_cost_bench PRIVATE
    Cellerator::prepared_relation_cuda cellerator_compute_sparse_project ${CMAKE_DL_LIBS})
