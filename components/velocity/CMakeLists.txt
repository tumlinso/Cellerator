include(GNUInstallDirs)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library(cellerator_velocity STATIC)
add_library(Cellerator::velocity ALIAS cellerator_velocity)
set_target_properties(cellerator_velocity PROPERTIES
  EXPORT_NAME velocity_binary CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_velocity PUBLIC cxx_std_20)
target_sources(cellerator_velocity
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS src FILES src/cellerator.velocity.cppm
  PRIVATE src/velocity.cpp src/precision.cpp src/detail/prepared.cpp
    src/detail/fit.cpp src/detail/neighborhood.cpp src/detail/stochastic_fit.cpp src/detail/cpu_kernels.cpp)
target_link_libraries(cellerator_velocity PUBLIC Cellerator::headers)
target_compile_options(cellerator_velocity PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
if(CELLERATOR_ENABLE_METHOD_SANITIZERS)
  target_compile_options(cellerator_velocity PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(cellerator_velocity PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS cellerator_velocity EXPORT CelleratorVelocityTargets
  FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_DATADIR}/Cellerator/modules"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(EXPORT CelleratorVelocityTargets NAMESPACE Cellerator::
  CXX_MODULES_DIRECTORY velocity-cxx-modules
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES VelocityComponent.cmake ../../cmake/CelleratorModuleInterfaces.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES
  "../../docs/licenses/ccell/LICENSE"
  "../../docs/licenses/velocity/THIRD_PARTY_NOTICES.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/cellerator/velocity")

if(CELLERATOR_BUILD_TESTS)
  find_package(Threads REQUIRED)
  foreach(suite IN ITEMS reference customization stochastic stochastic_customization
      cpu_kernels precision precision_cpu precision_promotion numeric_views fp16 blocks)
    add_executable(cellerator_velocity_${suite} tests/test_${suite}.cpp)
    target_link_libraries(cellerator_velocity_${suite} PRIVATE Cellerator::velocity)
    add_test(NAME cellerator-velocity-${suite} COMMAND cellerator_velocity_${suite})
  endforeach()
  target_link_libraries(cellerator_velocity_blocks PRIVATE Threads::Threads)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CELLERATOR_ENABLE_METHOD_SANITIZERS)
    add_executable(cellerator_velocity_allocations tests/test_allocations.cpp)
    target_link_libraries(cellerator_velocity_allocations PRIVATE Cellerator::velocity)
    add_test(NAME cellerator-velocity-allocations COMMAND cellerator_velocity_allocations)
    set_tests_properties(cellerator-velocity-allocations PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  foreach(example IN ITEMS steady_state stochastic selective_precision)
    add_executable(cellerator_velocity_${example}_example examples/${example}.cpp)
    target_link_libraries(cellerator_velocity_${example}_example PRIVATE Cellerator::velocity)
    add_test(NAME cellerator-velocity-${example}-example COMMAND cellerator_velocity_${example}_example)
  endforeach()
endif()

if(CELLERATOR_BUILD_METHOD_QUALIFICATION)
  add_executable(velocitypp_precision_benchmark benchmarks/precision_qualification.cpp)
  add_executable(velocitypp_sensitivity_driver tests/sensitivity_driver.cpp)
  add_executable(velocitypp_precision_interop_driver tests/precision_interop_driver.cpp)
  add_executable(velocitypp_interop_driver tests/interop_driver.cpp)
  add_executable(velocitypp_benchmark benchmarks/qualification.cpp)
  add_executable(velocitypp_stochastic_interop_driver tests/stochastic_interop_driver.cpp)
  add_executable(velocitypp_stochastic_benchmark benchmarks/stochastic_qualification.cpp)
  foreach(driver IN ITEMS precision_benchmark sensitivity_driver precision_interop_driver
      interop_driver benchmark stochastic_interop_driver stochastic_benchmark)
    target_link_libraries(velocitypp_${driver} PRIVATE Cellerator::velocity)
  endforeach()
endif()
