if(NOT ANNDATAPP_ENABLE_CUDA)
  return()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA target architectures")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "ANNDATAPP_ENABLE_CUDA requires an NVCC CUDA compiler")
endif()
enable_language(CUDA)
find_package(CUDAToolkit 12 REQUIRED)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
add_library(anndatapp_cuda)
add_library(anndatapp::cuda ALIAS anndatapp_cuda)
set_target_properties(anndatapp_cuda PROPERTIES EXPORT_NAME cuda_binary
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY None)
target_compile_features(anndatapp_cuda PUBLIC cxx_std_20)
target_sources(anndatapp_cuda PUBLIC FILE_SET CXX_MODULES
  BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES src/anndatapp.cuda.cppm
  PRIVATE src/cuda.cpp src/cuda/kernels.cu)
target_include_directories(anndatapp_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(anndatapp_cuda PUBLIC anndatapp::anndatapp PRIVATE CUDA::cudart)
target_compile_options(anndatapp_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--expt-relaxed-constexpr>)
install(TARGETS anndatapp_cuda EXPORT anndatappCudaTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/anndatapp/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT anndatappCudaTargets NAMESPACE anndatapp:: CXX_MODULES_DIRECTORY cuda-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndatapp)
install(FILES include/anndatapp/device_abi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/anndatapp)
if(ANNDATAPP_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda.cpp")
  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE anndatapp::cuda CUDA::cudart)
endif()
