add_executable(ce_nf1_b01 b01.cc)
target_link_libraries(ce_nf1_b01 PRIVATE Cellerator::native_foundation)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_nf1_b01 PRIVATE -Wall -Wextra -Werror -UNDEBUG)
endif()
add_test(NAME ce_nf1_b01 COMMAND ce_nf1_b01)
set_tests_properties(ce_nf1_b01 PROPERTIES LABELS "nf1;host;linked" TIMEOUT 30)

# Existing segment numerical test links the extracted production owners.
add_executable(ce_nf1_b02 "${_nf1_root}/src/compute/candidate/segment/reduce_v2_test.cc")
cellerator_link_native_foundation(ce_nf1_b02)
add_test(NAME ce_nf1_b02 COMMAND ce_nf1_b02)
set_tests_properties(ce_nf1_b02 PROPERTIES LABELS "nf1;host;linked;segment" TIMEOUT 30)
add_test(NAME ce_nf1_b02_boundary COMMAND "${CMAKE_COMMAND}"
    "-DREPO_ROOT=${_nf1_root}" "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/boundary"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/boundary.cmake")
set_tests_properties(ce_nf1_b02_boundary PROPERTIES LABELS "nf1;host;negative" TIMEOUT 30)
add_test(NAME ce_nf1_b03 COMMAND "${CMAKE_COMMAND}"
    "-DREPO_ROOT=${_nf1_root}" "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/toolchain"
    "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/toolchain.cmake")
set_tests_properties(ce_nf1_b03 PROPERTIES LABELS "nf1;host;toolchain" TIMEOUT 60)
