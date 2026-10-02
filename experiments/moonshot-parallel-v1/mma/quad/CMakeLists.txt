add_library(moonshot_quad STATIC quad_mma.cu)
target_include_directories(moonshot_quad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(moonshot_quad PROPERTIES CUDA_ARCHITECTURES 70 CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
add_executable(moonshot_quad_test quad_test.cu)
target_link_libraries(moonshot_quad_test PRIVATE moonshot_quad)
set_target_properties(moonshot_quad_test PROPERTIES CUDA_ARCHITECTURES 70 CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
# Host admission checks require no CUDA device; GPU execution is resource gated.
add_test(NAME moonshot_quad_admission COMMAND moonshot_quad_test --host)
