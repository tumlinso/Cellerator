get_filename_component(_ce_ops_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_op_matrix_patch STATIC patch.cc patch_provider.cc)
add_library(Cellerator::matrix_patch ALIAS cellerator_op_matrix_patch)
target_include_directories(cellerator_op_matrix_patch PUBLIC
    $<BUILD_INTERFACE:${_ce_ops_root}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(cellerator_op_matrix_patch PROPERTIES EXPORT_NAME matrix_patch)
target_compile_features(cellerator_op_matrix_patch PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_op_matrix_patch PRIVATE -fno-fast-math -ffp-contract=off)
endif()

add_library(cellerator_narrow_matmul STATIC narrow_matmul.cc)
add_library(Cellerator::narrow_matmul ALIAS cellerator_narrow_matmul)
target_include_directories(cellerator_narrow_matmul PUBLIC
    $<BUILD_INTERFACE:${_ce_ops_root}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_narrow_matmul PUBLIC cxx_std_20)
target_compile_options(cellerator_narrow_matmul PRIVATE -fno-fast-math -ffp-contract=off)
set_target_properties(cellerator_narrow_matmul PROPERTIES EXPORT_NAME narrow_matmul)
install(TARGETS cellerator_narrow_matmul EXPORT CelleratorCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_narrow_numeric_test
        ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL} "${_ce_ops_root}/tests/math_reference/narrow_numeric_test.cc")
    target_link_libraries(cellerator_narrow_numeric_test PRIVATE Cellerator::narrow_matmul)
    add_test(NAME cellerator_narrow_numeric COMMAND cellerator_narrow_numeric_test)
endif()

if(CELLERATOR_HAS_CUDA)
    if(CUDAToolkit_VERSION VERSION_LESS 13
       AND CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)70(-real|-virtual)?(;|$)")
        add_library(cellerator_tiled_matmul_cuda STATIC tiled_matmul_cuda.cu)
        add_library(Cellerator::tiled_matmul_cuda ALIAS cellerator_tiled_matmul_cuda)
        target_link_libraries(cellerator_tiled_matmul_cuda PUBLIC Cellerator::headers CUDA::cudart)
        set_target_properties(cellerator_tiled_matmul_cuda PROPERTIES
            EXPORT_NAME tiled_matmul_cuda CUDA_ARCHITECTURES 70-real
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        install(TARGETS cellerator_tiled_matmul_cuda EXPORT CelleratorCoreTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

        if(CELLERATOR_BUILD_TESTS)
            add_executable(cellerator_tiled_matmul_volta_test
                "${_ce_ops_root}/tests/tensor_core/sm70/tiled_matmul_test.cu")
            target_link_libraries(cellerator_tiled_matmul_volta_test PRIVATE
                Cellerator::tiled_matmul_cuda Cellerator::memory_cuda)
            set_target_properties(cellerator_tiled_matmul_volta_test PROPERTIES
                CUDA_ARCHITECTURES 70-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
            add_test(NAME cellerator_tiled_matmul_volta COMMAND cellerator_tiled_matmul_volta_test)
            set_tests_properties(cellerator_tiled_matmul_volta PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()

    add_library(cellerator_narrow_matmul_cuda STATIC narrow_matmul_cuda.cu)
    add_library(Cellerator::narrow_matmul_cuda ALIAS cellerator_narrow_matmul_cuda)
    target_link_libraries(cellerator_narrow_matmul_cuda PUBLIC
        Cellerator::narrow_matmul CUDA::cublasLt CUDA::cudart)
    set_target_properties(cellerator_narrow_matmul_cuda PROPERTIES
        EXPORT_NAME narrow_matmul_cuda CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    install(TARGETS cellerator_narrow_matmul_cuda EXPORT CelleratorCoreTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    add_library(cellerator_csr_multiply STATIC csr_spmm.cu csr_spmv.cu csr_cusparse.cu)
    add_library(Cellerator::csr_multiply ALIAS cellerator_csr_multiply)
    target_include_directories(cellerator_csr_multiply
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        PRIVATE ${_ce_ops_root}/include)
    target_link_libraries(cellerator_csr_multiply PRIVATE CUDA::cudart CUDA::cusparse)
    set_target_properties(cellerator_csr_multiply PROPERTIES
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
endif()
unset(_ce_ops_root)
