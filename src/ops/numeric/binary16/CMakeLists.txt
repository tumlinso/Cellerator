cmake_minimum_required(VERSION 3.28)
project(CelleratorBinary16 VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT TARGET Cellerator::binary16)
  add_library(CelleratorBinary16 INTERFACE)
  add_library(Cellerator::binary16 ALIAS CelleratorBinary16)
  set_target_properties(CelleratorBinary16 PROPERTIES EXPORT_NAME binary16)
  target_compile_features(CelleratorBinary16 INTERFACE cxx_std_20)
  target_include_directories(CelleratorBinary16 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  install(TARGETS CelleratorBinary16 EXPORT CelleratorBinary16Targets)
  install(EXPORT CelleratorBinary16Targets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorBinary16)
  install(FILES ${CMAKE_CURRENT_LIST_DIR}/../../../../include/Cellerator/ops/numeric/binary16.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/ops/numeric)

  configure_package_config_file(cmake/CelleratorBinary16Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorBinary16Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorBinary16)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/CelleratorBinary16ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorBinary16Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CelleratorBinary16ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleratorBinary16)
endif()

option(CELLERATOR_BINARY16_BUILD_TESTS "Build binary16 conversion tests" ${PROJECT_IS_TOP_LEVEL})
if(CELLERATOR_BINARY16_BUILD_TESTS)
  enable_testing()
  add_executable(test_binary16 tests/test_binary16.cpp)
  target_link_libraries(test_binary16 PRIVATE Cellerator::binary16)
  add_test(NAME binary16 COMMAND test_binary16)

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
      RESULT_VARIABLE CELLERATOR_BINARY16_NUMPY_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(CELLERATOR_BINARY16_NUMPY_RESULT EQUAL 0)
      add_test(NAME binary16_numpy_oracle COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/tests/python_float16.py" $<TARGET_FILE:test_binary16>)
    endif()
  endif()
endif()
