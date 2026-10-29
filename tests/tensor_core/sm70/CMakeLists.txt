function(cellerator_add_sm70_vertical_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        Cellerator::architecture_provider
        CUDA::cudart
    )
    # These inherited regression sources use assert as their fail-closed test
    # primitive; keep assertions active under a Release integration build.
    target_compile_options(${target} PRIVATE -UNDEBUG)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

cellerator_add_sm70_vertical_test(geometry_sm70_provider_test
    tests/tensor_core/sm70/provider_test.cu)
cellerator_add_sm70_vertical_test(geometry_sm70_value_pack_test
    tests/tensor_core/sm70/value_pack_test.cu)
cellerator_add_sm70_vertical_test(geometry_sm70_n64_test
    tests/tensor_core/sm70/relation_apply_n64_test.cu)
cellerator_add_sm70_vertical_test(geometry_sm70_hybrid_test
    tests/tensor_core/sm70/relation_apply_hybrid_test.cu)

add_custom_target(geometry_sm70_vertical_tests DEPENDS
    geometry_sm70_provider_test
    geometry_sm70_value_pack_test
    geometry_sm70_n64_test
    geometry_sm70_hybrid_test
)
