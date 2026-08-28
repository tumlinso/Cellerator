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

# CE-LIVE-43: the native adapter surface is split by responsibility.  These
# targets share Torch build policy but retain Cellerator ownership of storage,
# planning, execution, parameters, and readiness.
add_library(cellera_torch_native_views STATIC
    src/native_views.cc
)
add_library(CelleraTorch::native_views ALIAS cellera_torch_native_views)
target_include_directories(cellera_torch_native_views PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_native_views PUBLIC
    CelleraTorch::torch
    Cellerator::runtime
)
target_compile_features(cellera_torch_native_views PUBLIC cxx_std_17)

add_library(cellera_torch_program_ops STATIC
    src/program_ops.cu
)
add_library(CelleraTorch::program_ops ALIAS cellera_torch_program_ops)
target_include_directories(cellera_torch_program_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_program_ops PUBLIC
    CelleraTorch::torch
    Cellerator::executable_program
)
set_target_properties(cellera_torch_program_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellera_torch_autograd_ops STATIC
    src/autograd_ops.cu
)
add_library(CelleraTorch::autograd_ops ALIAS cellera_torch_autograd_ops)
target_include_directories(cellera_torch_autograd_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cellera_torch_autograd_ops PUBLIC
    CelleraTorch::torch
    Cellerator::training_program
)
set_target_properties(cellera_torch_autograd_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellera_torch_native_adapter INTERFACE)
add_library(CelleraTorch::native_adapter ALIAS cellera_torch_native_adapter)
target_link_libraries(cellera_torch_native_adapter INTERFACE
    CelleraTorch::native_views
    CelleraTorch::program_ops
    CelleraTorch::autograd_ops
)

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
    Cellerator::runtime_fleet
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

add_executable(celleraTorchNativeViewsTest
    tests/native_views_test.cc
)
target_link_libraries(celleraTorchNativeViewsTest PRIVATE
    CelleraTorch::native_views
)
set_target_properties(celleraTorchNativeViewsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)

add_executable(celleraTorchProgramOpsTest
    tests/program_ops_test.cc
)
target_link_libraries(celleraTorchProgramOpsTest PRIVATE
    CelleraTorch::program_ops
)
set_target_properties(celleraTorchProgramOpsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)

add_executable(celleraTorchAutogradOpsTest
    tests/autograd_ops_test.cc
)
target_link_libraries(celleraTorchAutogradOpsTest PRIVATE
    CelleraTorch::autograd_ops
)
set_target_properties(celleraTorchAutogradOpsTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
)

add_custom_target(celleraTorchNativeAdapterTests
    DEPENDS
        celleraTorchNativeViewsTest
        celleraTorchProgramOpsTest
        celleraTorchAutogradOpsTest
)

add_custom_target(celleraTorchNativeAdapterCheck
    COMMAND $<TARGET_FILE:celleraTorchNativeViewsTest>
    COMMAND $<TARGET_FILE:celleraTorchProgramOpsTest>
    COMMAND $<TARGET_FILE:celleraTorchAutogradOpsTest>
    DEPENDS celleraTorchNativeAdapterTests
    USES_TERMINAL
)

# CE-LIVE-44 owns the quantitative source and evidence.  The Wave D fan-in
# owns its shared build seam, which becomes active when that scoped source is
# present and CMake is regenerated.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantitative_smoke_test.cc")
    add_executable(celleraTorchQuantitativeSmokeTest
        tests/quantitative_smoke_test.cc
    )
    target_link_libraries(celleraTorchQuantitativeSmokeTest PRIVATE
        CelleraTorch::native_adapter
        Cellerator::identity_registry
        cellerator_live_runtime_fixture
    )
    target_include_directories(celleraTorchQuantitativeSmokeTest PRIVATE
        ${CMAKE_SOURCE_DIR}
    )
    set_target_properties(celleraTorchQuantitativeSmokeTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY "${CELLERATORCH_TEST_OUTPUT_DIR}"
    )
endif()
