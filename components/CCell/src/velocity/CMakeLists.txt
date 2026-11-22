add_library(ccell_velocity)
add_library(ccell::velocity ALIAS ccell_velocity)
set_target_properties(ccell_velocity PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME velocity_binary)
target_compile_features(ccell_velocity PUBLIC cxx_std_20)
target_sources(ccell_velocity PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${ccell_SOURCE_DIR}/src"
  FILES "${ccell_SOURCE_DIR}/src/ccell.velocity.cppm" src/velocitypp.cppm src/velocitypp.precision.cppm
  PRIVATE src/velocitypp.cpp src/detail/prepared.cpp src/detail/fit.cpp src/detail/neighborhood.cpp
    src/detail/stochastic_fit.cpp src/detail/cpu_kernels.cpp src/precision.cpp)
target_include_directories(ccell_velocity PUBLIC
  $<BUILD_INTERFACE:${ccell_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ccell_velocity PUBLIC Cellerator::headers)
target_compile_options(ccell_velocity PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

install(TARGETS ccell_velocity EXPORT ccellMethodTargets
  FILE_SET CXX_MODULES DESTINATION "${CCELL_INSTALL_MODULE_DIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

if(CCELL_BUILD_TESTS)
  foreach(suite IN ITEMS reference customization stochastic stochastic_customization
      cpu_kernels precision precision_cpu precision_promotion numeric_views)
    add_executable(ccell_velocity_${suite} tests/test_${suite}.cpp)
    target_link_libraries(ccell_velocity_${suite} PRIVATE ccell::velocity)
    add_test(NAME ccell-velocity-${suite} COMMAND ccell_velocity_${suite})
  endforeach()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CCELL_ENABLE_SANITIZERS)
    add_executable(ccell_velocity_allocations tests/test_allocations.cpp)
    target_link_libraries(ccell_velocity_allocations PRIVATE ccell::velocity)
    add_test(NAME ccell-velocity-allocations COMMAND ccell_velocity_allocations)
    set_tests_properties(ccell-velocity-allocations PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  foreach(example IN ITEMS steady_state stochastic selective_precision)
    add_executable(ccell_velocity_${example}_example examples/${example}.cpp)
    target_link_libraries(ccell_velocity_${example}_example PRIVATE ccell::velocity)
    add_test(NAME ccell-velocity-${example}-example COMMAND ccell_velocity_${example}_example)
  endforeach()
endif()

if(CCELL_BUILD_QUALIFICATION)
  add_executable(velocitypp_precision_benchmark benchmarks/precision_qualification.cpp)
  add_executable(velocitypp_sensitivity_driver tests/sensitivity_driver.cpp)
  add_executable(velocitypp_precision_interop_driver tests/precision_interop_driver.cpp)
  add_executable(velocitypp_interop_driver tests/interop_driver.cpp)
  add_executable(velocitypp_benchmark benchmarks/qualification.cpp)
  add_executable(velocitypp_stochastic_interop_driver tests/stochastic_interop_driver.cpp)
  add_executable(velocitypp_stochastic_benchmark benchmarks/stochastic_qualification.cpp)
  foreach(driver IN ITEMS precision_benchmark sensitivity_driver precision_interop_driver
      interop_driver benchmark stochastic_interop_driver stochastic_benchmark)
    target_link_libraries(velocitypp_${driver} PRIVATE ccell::velocity)
  endforeach()
endif()
