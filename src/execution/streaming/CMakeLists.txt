add_library(cellerator_execution_streaming STATIC fetch.cc)
add_library(Cellerator::execution_streaming ALIAS cellerator_execution_streaming)
set_target_properties(cellerator_execution_streaming PROPERTIES EXPORT_NAME execution_streaming)
target_include_directories(cellerator_execution_streaming PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_execution_streaming PUBLIC cxx_std_20)
if(CELLERATOR_HAS_CUDA)
    target_link_libraries(cellerator_execution_streaming PUBLIC CUDA::cudart)
else()
    target_compile_definitions(cellerator_execution_streaming PUBLIC CELLERATOR_ENABLE_CUDA=0)
endif()
install(TARGETS cellerator_execution_streaming EXPORT CelleratorCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_fetch_test
        ${PROJECT_SOURCE_DIR}/tests/execution/streaming/fetch_test.cc)
    target_link_libraries(cellerator_fetch_test PRIVATE Cellerator::execution_streaming)
    add_test(NAME cellerator_fetch COMMAND cellerator_fetch_test)

    add_executable(cellerator_sliced_ell_pack_test
        ${PROJECT_SOURCE_DIR}/tests/layout/sliced_ell_pack_test.cc)
    target_link_libraries(cellerator_sliced_ell_pack_test PRIVATE Cellerator::headers)
    target_compile_definitions(cellerator_sliced_ell_pack_test PRIVATE CELLERATOR_ENABLE_CUDA=0)
    add_test(NAME cellerator_sliced_ell_pack COMMAND cellerator_sliced_ell_pack_test)
endif()

if(TARGET Cellerator::memory_cuda)
    add_library(cellerator_execution_streaming_cuda STATIC cuda_staging.cu)
    add_library(Cellerator::execution_streaming_cuda ALIAS cellerator_execution_streaming_cuda)
    set_target_properties(cellerator_execution_streaming_cuda PROPERTIES
        EXPORT_NAME execution_streaming_cuda
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    target_link_libraries(cellerator_execution_streaming_cuda PUBLIC
        Cellerator::execution_streaming Cellerator::memory_cuda)
    install(TARGETS cellerator_execution_streaming_cuda EXPORT CelleratorExecutionStreamingCUDATargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT CelleratorExecutionStreamingCUDATargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorExecutionStreamingCUDATargets.cmake\")\n")

    if(CELLERATOR_BUILD_TESTS)
        add_executable(cellerator_fetch_cuda_test
            ${PROJECT_SOURCE_DIR}/tests/execution/streaming/cuda_staging_test.cu)
        target_link_libraries(cellerator_fetch_cuda_test PRIVATE
            Cellerator::execution_streaming_cuda cellerator_compute_sparse_ops)
        set_target_properties(cellerator_fetch_cuda_test PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME cellerator_fetch_cuda COMMAND cellerator_fetch_cuda_test)
        set_tests_properties(cellerator_fetch_cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
