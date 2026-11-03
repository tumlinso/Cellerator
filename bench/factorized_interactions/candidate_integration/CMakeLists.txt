if(NOT TARGET Cellerator::factorized_interaction_cuda)
    message(FATAL_ERROR
        "This subdirectory must be attached to the source-root Cellerator build after its native CUDA target is defined")
endif()

if(NOT DEFINED CELLERATOR_SOURCE_ROOT OR
   NOT EXISTS "${CELLERATOR_SOURCE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "CELLERATOR_SOURCE_ROOT must name the source snapshot")
endif()

set(_candidate_runner
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/candidate_integration/candidate_runner.cu")
set(_minimal_direct_source
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/controls/minimal_direct_layout.cu")
set(_unique_output_source
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/candidates/direct_unique_output/direct_unique_output.cu")
foreach(_required IN ITEMS _candidate_runner _minimal_direct_source _unique_output_source)
    if(NOT EXISTS "${${_required}}")
        message(FATAL_ERROR "Candidate runner source is missing: ${${_required}}")
    endif()
endforeach()

add_executable(cellerator_factorized_interaction_candidate_runner
    "${_candidate_runner}"
    "${_minimal_direct_source}"
    "${_unique_output_source}")
target_compile_features(cellerator_factorized_interaction_candidate_runner PRIVATE cxx_std_17)
set_target_properties(cellerator_factorized_interaction_candidate_runner PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(cellerator_factorized_interaction_candidate_runner PRIVATE
    "${CELLERATOR_SOURCE_ROOT}/include"
    "${CELLERATOR_SOURCE_ROOT}/src/modeling/cuda"
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/controls"
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/candidates/direct_unique_output")
target_link_libraries(cellerator_factorized_interaction_candidate_runner PRIVATE
    Cellerator::factorized_interaction_cuda CUDA::cudart)

add_executable(cellerator_factorized_interaction_baseline_runner
    "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/native_runner.cu")
target_compile_features(cellerator_factorized_interaction_baseline_runner PRIVATE cxx_std_17)
set_target_properties(cellerator_factorized_interaction_baseline_runner PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_link_libraries(cellerator_factorized_interaction_baseline_runner PRIVATE
    Cellerator::factorized_interaction_cuda CUDA::cudart)

