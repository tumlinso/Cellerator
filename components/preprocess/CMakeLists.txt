include(GNUInstallDirs)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library(cellerator_preprocess STATIC)
add_library(Cellerator::preprocess ALIAS cellerator_preprocess)
set_target_properties(cellerator_preprocess PROPERTIES
  EXPORT_NAME preprocess_binary CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_preprocess PUBLIC cxx_std_20)
target_sources(cellerator_preprocess
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS src FILES src/cellerator.preprocess.cppm src/qc.cppm src/hvg.cppm
  PRIVATE src/preprocess.cpp src/qc.cpp src/hvg.cpp)
target_link_libraries(cellerator_preprocess PUBLIC Cellerator::headers)
target_compile_options(cellerator_preprocess PRIVATE -Wall -Wextra -Wpedantic)
if(CELLERATOR_ENABLE_METHOD_SANITIZERS)
  target_compile_options(cellerator_preprocess PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(cellerator_preprocess PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS cellerator_preprocess EXPORT CelleratorPreprocessTargets
  FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_DATADIR}/Cellerator/modules"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(EXPORT CelleratorPreprocessTargets NAMESPACE Cellerator::
  CXX_MODULES_DIRECTORY preprocess-cxx-modules
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES PreprocessComponent.cmake ../../cmake/CelleratorModuleInterfaces.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES
  "../../docs/licenses/ccell/LICENSE"
  "../../docs/licenses/ccell/THIRD_PARTY_NOTICES.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/cellerator/preprocess")

if(CELLERATOR_BUILD_TESTS)
  foreach(suite IN ITEMS numerics hvg dispatch transform)
    add_executable(cellerator_preprocess_${suite} tests/test_${suite}.cpp)
    target_link_libraries(cellerator_preprocess_${suite} PRIVATE Cellerator::preprocess)
    add_test(NAME cellerator-preprocess-${suite} COMMAND cellerator_preprocess_${suite})
  endforeach()
  add_executable(cellerator_preprocess_zeroalloc tests/test_zeroalloc.cpp)
  target_link_libraries(cellerator_preprocess_zeroalloc PRIVATE Cellerator::preprocess)
  if(CELLERATOR_ENABLE_METHOD_SANITIZERS)
    add_test(NAME cellerator-preprocess-sanitized-behavior COMMAND cellerator_preprocess_zeroalloc)
    set_tests_properties(cellerator-preprocess-sanitized-behavior PROPERTIES SKIP_RETURN_CODE 77)
  else()
    add_test(NAME cellerator-preprocess-zeroalloc COMMAND cellerator_preprocess_zeroalloc)
    set_tests_properties(cellerator-preprocess-zeroalloc PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  add_executable(cellerator_preprocess_example examples/basic.cpp)
  target_link_libraries(cellerator_preprocess_example PRIVATE Cellerator::preprocess)
  add_test(NAME cellerator-preprocess-example COMMAND cellerator_preprocess_example)
endif()

if(CELLERATOR_BUILD_TESTS)
  add_executable(cellerator_preprocess_modeling_example examples/modeling.cpp)
  target_link_libraries(cellerator_preprocess_modeling_example PRIVATE Cellerator::preprocess)
  add_test(NAME cellerator-preprocess-modeling COMMAND cellerator_preprocess_modeling_example)
endif()

if(CELLERATOR_BUILD_METHOD_QUALIFICATION)
  if(TARGET anndatapp::anndatapp)
    add_executable(cellerator_preprocess_interop_driver tests/interop_driver.cpp)
    target_link_libraries(cellerator_preprocess_interop_driver PRIVATE Cellerator::preprocess anndatapp::anndatapp)
  endif()
  add_executable(cellerator_preprocess_benchmark benchmarks/qualification.cpp)
  target_link_libraries(cellerator_preprocess_benchmark PRIVATE Cellerator::preprocess)
endif()
