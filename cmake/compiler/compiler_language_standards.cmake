include_guard(GLOBAL)

function(cellerator_set_compiler_implementation_standard target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 23 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
endfunction()

function(cellerator_set_legacy_accelerator_standard target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endfunction()
