# Independent product2 operation: existing indexed mechanism ABI is unchanged.
option(CELLERATOR_BUILD_PRODUCT2 "Build standalone prepared product2 operation" OFF)
if(CELLERATOR_BUILD_PRODUCT2)
  add_library(cellerator_product2 SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/product2/product2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/execution/program/program_v2.cc)
  add_library(Cellerator::product2 ALIAS cellerator_product2)
  target_compile_features(cellerator_product2 PUBLIC cxx_std_20)
  target_include_directories(cellerator_product2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF")
    enable_language(CUDA)
    target_sources(cellerator_product2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/operation/product2/product2.cu)
    target_link_libraries(cellerator_product2 PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(cellerator_product2 PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(cellerator_product2 PUBLIC CELLERATOR_PRODUCT2_HAS_CUDA=1)
    if(CELLERATOR_BUILD_TESTS)
      enable_testing()
      add_executable(cellerator_product2_cuda_smoke ${CMAKE_CURRENT_SOURCE_DIR}/experiments/moonshot-parallel-v1/integration/product2_cuda_smoke.cu)
      target_link_libraries(cellerator_product2_cuda_smoke PRIVATE Cellerator::product2 CUDA::cudart CUDA::cuda_driver)
      set_target_properties(cellerator_product2_cuda_smoke PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()
  endif()
  if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_executable(cellerator_product2_test ${CMAKE_CURRENT_SOURCE_DIR}/experiments/moonshot-parallel-v1/integration/native_product2_test.cc)
    target_link_libraries(cellerator_product2_test PRIVATE Cellerator::product2)
    add_test(NAME cellerator_product2_native COMMAND cellerator_product2_test)
  endif()
endif()
if(CELLERATOR_BUILD_PRODUCT2)
  set_target_properties(cellerator_product2 PROPERTIES EXPORT_NAME product2)
  install(TARGETS cellerator_product2 EXPORT CelleratorProduct2Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Product2
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Product2
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Product2)
  # Carry the public operation's established identity/program dependency headers
  # in the independently installable component, without changing native libraries.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Product2)
  install(EXPORT CelleratorProduct2Targets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorProduct2 COMPONENT Product2)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CelleratorProduct2Config.cmake
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorProduct2Targets.cmake\")\n")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CelleratorProduct2Config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorProduct2 COMPONENT Product2)
endif()
