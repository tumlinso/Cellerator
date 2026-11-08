if(NOT VELOCITY_ENABLE_CUDA)
  return()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA target architectures")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "VELOCITY_ENABLE_CUDA requires an NVCC CUDA compiler")
endif()
enable_language(CUDA)
find_package(CUDAToolkit 12 REQUIRED)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
add_library(velocity_cuda)
add_library(velocity::cuda ALIAS velocity_cuda)
set_target_properties(velocity_cuda PROPERTIES EXPORT_NAME cuda_binary
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY None)
target_compile_features(velocity_cuda PUBLIC cxx_std_20)
target_sources(velocity_cuda PUBLIC FILE_SET CXX_MODULES
  BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES src/velocity.cuda.cppm
  PRIVATE src/cuda.cpp src/cuda/kernels.cu src/cuda/sensitivity.cu)
target_link_libraries(velocity_cuda PUBLIC velocity::velocity anndata::cuda PRIVATE CUDA::cudart)
target_compile_options(velocity_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--expt-relaxed-constexpr>)
install(TARGETS velocity_cuda EXPORT velocityCudaTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/velocity/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT velocityCudaTargets NAMESPACE velocity:: CXX_MODULES_DIRECTORY cuda-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocity)
if(VELOCITY_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda.cpp")
  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE velocity::cuda CUDA::cudart)
endif()

if(VELOCITY_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_precision_cuda.cpp")
  add_executable(test_precision_cuda tests/test_precision_cuda.cpp)
  target_link_libraries(test_precision_cuda PRIVATE velocity::cuda CUDA::cudart)
endif()

if(VELOCITY_BUILD_TESTS)
  add_test(NAME installed-cuda-build COMMAND ${CMAKE_COMMAND}
    -DVELOCITY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DVELOCITY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DVELOCITY_COMPILER=${CMAKE_CXX_COMPILER}
    -DVELOCITY_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DVELOCITY_CONFIG=$<CONFIG>
    -DVELOCITY_ANNDATA_PREFIX=${anndata_DIR}/../../..
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-cuda-consumer.cmake)
endif()
if(VELOCITY_BUILD_QUALIFICATION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/precision_cuda_qualification.cpp")
  add_executable(velocity_precision_cuda_benchmark benchmarks/precision_cuda_qualification.cpp)
  target_link_libraries(velocity_precision_cuda_benchmark PRIVATE velocity::cuda CUDA::cudart ${CMAKE_DL_LIBS})
endif()
if(VELOCITY_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_batches.cpp")
  add_executable(test_cuda_batches tests/test_cuda_batches.cpp)
  target_link_libraries(test_cuda_batches PRIVATE velocity::cuda CUDA::cudart)
endif()
