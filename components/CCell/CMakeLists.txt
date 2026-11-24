project(ccell VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CCELL_ENABLE_SANITIZERS "Instrument CCell reference methods and consumers" OFF)
option(CCELL_BUILD_QUALIFICATION "Build CCell numerical qualification drivers" OFF)
set(CCELL_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/ccell")
set(CCELL_INSTALL_MODULE_DIR "${CMAKE_INSTALL_DATADIR}/ccell/modules")

add_subdirectory(src/preprocess)
add_subdirectory(src/velocity)
add_library(ccell)
add_library(ccell::ccell ALIAS ccell)
set_target_properties(ccell PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME ccell_binary)
target_compile_features(ccell PUBLIC cxx_std_20)
target_sources(ccell PUBLIC FILE_SET CXX_MODULES BASE_DIRS src FILES src/ccell.cppm)
target_link_libraries(ccell PUBLIC ccell::preprocess ccell::velocity)

if(CCELL_ENABLE_SANITIZERS)
  foreach(target IN ITEMS ccell_preprocess ccell_velocity ccell)
    if(TARGET ${target})
      target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endif()
  endforeach()
endif()

install(TARGETS ccell EXPORT ccellMethodTargets
  FILE_SET CXX_MODULES DESTINATION "${CCELL_INSTALL_MODULE_DIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(EXPORT ccellMethodTargets NAMESPACE ccell:: CXX_MODULES_DIRECTORY method-cxx-modules
  DESTINATION "${CCELL_INSTALL_PACKAGE_DIR}")
configure_package_config_file(cmake/ccellConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/ccellConfig.cmake"
  INSTALL_DESTINATION "${CCELL_INSTALL_PACKAGE_DIR}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ccellConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ccellConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ccellConfigVersion.cmake" cmake/ccellModuleProvider.cmake
  DESTINATION "${CCELL_INSTALL_PACKAGE_DIR}")
install(DIRECTORY include/CCell DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ccell")
install(FILES src/velocity/THIRD_PARTY_NOTICES.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ccell/velocity")

if(CCELL_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME ccell-native-style COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_native_style.py"
      --source src/ccell.cppm --source src/ccell.preprocess.cppm
      --source src/ccell.velocity.cppm
      --source src/preprocess --source src/velocity/src
      --source examples --source src/velocity/examples --source include)
    add_test(NAME ccell-native-style-fixtures COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/style/test_native_style.py")
  endif()
  foreach(suite IN ITEMS umbrella components)
    add_executable(ccell_${suite} tests/${suite}.cpp)
    target_link_libraries(ccell_${suite} PRIVATE ccell::ccell)
    add_test(NAME ccell-${suite} COMMAND ccell_${suite})
  endforeach()

  add_executable(ccell_modeling_example examples/modeling.cpp)
  target_link_libraries(ccell_modeling_example PRIVATE ccell::preprocess)
  add_test(NAME ccell-modeling-example COMMAND ccell_modeling_example)

  set(CCELL_TEST_ANNDATA_MODELING OFF)
  if(TARGET Cellerator::anndata AND (ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR))
    set(CCELL_TEST_ANNDATA_MODELING ON)
    add_executable(ccell_anndata_modeling_test tests/anndata_modeling.cpp)
    target_link_libraries(ccell_anndata_modeling_test PRIVATE Cellerator::anndata ccell::preprocess ccell::velocity)
    add_test(NAME ccell-anndata-modeling COMMAND ccell_anndata_modeling_test)
  endif()

  add_test(NAME ccell-installed-consumer COMMAND "${CMAKE_COMMAND}"
    "-DCCELL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DCCELL_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DCELLERATOR_BUILD_DIR=${CMAKE_BINARY_DIR}"
    "-DCCELL_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCCELL_SCAN_DEPS=${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}"
    "-DCCELL_CONFIG=$<CONFIG>"
    "-DCCELL_TEST_ANNDATA_MODELING=${CCELL_TEST_ANNDATA_MODELING}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake")
endif()
