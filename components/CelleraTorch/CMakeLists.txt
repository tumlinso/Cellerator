set(CELLERATORCH_PREFERRED_LIBTORCH_ROOT "/usr/local" CACHE PATH
    "Preferred root for the source-built libtorch installation")

if(DEFINED ENV{LIBTORCH_PATH} AND NOT "$ENV{LIBTORCH_PATH}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
elseif(EXISTS "${CELLERATORCH_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    if(NOT Torch_DIR OR Torch_DIR MATCHES ".*/site-packages/torch/share/cmake/Torch/?$")
        set(Torch_DIR
            "${CELLERATORCH_PREFERRED_LIBTORCH_ROOT}/share/cmake/Torch"
            CACHE PATH "Torch CMake package directory" FORCE)
        message(STATUS "Defaulting Torch_DIR to source-built libtorch at ${Torch_DIR}")
    endif()
    if(EXISTS "${CELLERATORCH_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake")
        if(NOT Caffe2_DIR OR Caffe2_DIR MATCHES ".*/site-packages/torch/share/cmake/Caffe2/?$")
            set(Caffe2_DIR
                "${CELLERATORCH_PREFERRED_LIBTORCH_ROOT}/share/cmake/Caffe2"
                CACHE PATH "Caffe2 CMake package directory" FORCE)
            message(STATUS "Defaulting Caffe2_DIR to source-built libtorch at ${Caffe2_DIR}")
        endif()
    endif()
    foreach(celleratorch_torch_cache_var
            IN ITEMS
                TORCH_LIBRARY
                C10_CUDA_LIBRARY
                c10_LIBRARY
                kineto_LIBRARY)
        if(DEFINED ${celleratorch_torch_cache_var}
           AND "${${celleratorch_torch_cache_var}}" MATCHES ".*/site-packages/torch/.*")
            unset(${celleratorch_torch_cache_var} CACHE)
        endif()
    endforeach()
endif()

find_package(Torch REQUIRED)

set(CELLERATORCH_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}")

add_library(cellera_torch INTERFACE)
add_library(CelleraTorch::torch ALIAS cellera_torch)
target_link_libraries(cellera_torch INTERFACE
    CUDA::cudart
    ${TORCH_LIBRARIES}
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
)
target_compile_features(cellera_torch INTERFACE cxx_std_17)
if(DEFINED TORCH_INCLUDE_DIRS)
    target_include_directories(cellera_torch SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
endif()
if(TORCH_CXX_FLAGS)
    separate_arguments(CELLERATORCH_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(cellera_torch INTERFACE ${CELLERATORCH_TORCH_CXX_FLAGS})
endif()

add_library(cellera_torch_bindings INTERFACE)
add_library(CelleraTorch::bindings ALIAS cellera_torch_bindings)
target_include_directories(cellera_torch_bindings INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_bindings INTERFACE CelleraTorch::torch CellShard::runtime)
target_compile_features(cellera_torch_bindings INTERFACE cxx_std_17)

add_library(cellera_torch_model_ops STATIC
    src/model_ops/model_ops.cu
)
add_library(CelleraTorch::model_ops ALIAS cellera_torch_model_ops)
target_include_directories(cellera_torch_model_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_model_ops PUBLIC CelleraTorch::torch)
target_link_libraries(cellera_torch_model_ops PRIVATE CUDA::cudart)
set_target_properties(cellera_torch_model_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellera_torch_model_ops)

add_library(cellera_torch_quantize STATIC
    src/quantize/quantize_cuda.cu
)
add_library(CelleraTorch::quantize ALIAS cellera_torch_quantize)
target_include_directories(cellera_torch_quantize PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_quantize PUBLIC CelleraTorch::torch)
target_link_libraries(cellera_torch_quantize PRIVATE
    cellerator_compute_sparse_ops
    cellerator_compute_runtime
    CUDA::cudart
)
set_target_properties(cellera_torch_quantize PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellera_torch_quantize)

add_library(cellera_torch_dense_reduce INTERFACE)
add_library(CelleraTorch::dense_reduce ALIAS cellera_torch_dense_reduce)
target_include_directories(cellera_torch_dense_reduce INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_dense_reduce INTERFACE
    CelleraTorch::bindings
    CelleraTorch::model_ops
    CellShard::runtime
)
target_compile_features(cellera_torch_dense_reduce INTERFACE cxx_std_17)

add_executable(celleraTorchBindingsCompileTest
    tests/bindings_compile_test.cc
)
target_link_libraries(celleraTorchBindingsCompileTest PRIVATE CelleraTorch::bindings)
set_target_properties(celleraTorchBindingsCompileTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)
cellerator_enable_perf(celleraTorchBindingsCompileTest)

add_executable(celleraTorchDenseReduceCompileTest
    tests/dense_reduce_compile_test.cc
)
target_link_libraries(celleraTorchDenseReduceCompileTest PRIVATE CelleraTorch::dense_reduce)
set_target_properties(celleraTorchDenseReduceCompileTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)
cellerator_enable_perf(celleraTorchDenseReduceCompileTest)

add_executable(celleraTorchQuantizePrimitiveTest
    tests/quantize_primitive_test.cc
)
target_link_libraries(celleraTorchQuantizePrimitiveTest PRIVATE
    CelleraTorch::quantize
    CelleraTorch::model_ops
    CUDA::cudart
)
set_target_properties(celleraTorchQuantizePrimitiveTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)
cellerator_enable_perf(celleraTorchQuantizePrimitiveTest)

add_executable(celleraTorchModelCustomOpsTest
    tests/model_custom_ops_test.cc
)
target_link_libraries(celleraTorchModelCustomOpsTest PRIVATE
    CelleraTorch::dense_reduce
    CelleraTorch::model_ops
    CellShard::runtime
)
set_target_properties(celleraTorchModelCustomOpsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)
cellerator_enable_perf(celleraTorchModelCustomOpsTest)
