include_guard(GLOBAL)

function(cellerator_create_accelerator_compiler_smoke_target)
    if(NOT CELLERATOR_HAS_CUDA)
        return()
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_executable(cellerator_compiler_smoke_accelerator
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../tests/compiler/b01/create_accelerator_enabled_compiler_smoke_targets_test.cc")
    target_link_libraries(cellerator_compiler_smoke_accelerator PRIVATE
        Cellerator::CompilerRealization CUDA::cudart)
    if(TARGET Cellerator::runtime)
        target_link_libraries(cellerator_compiler_smoke_accelerator PRIVATE Cellerator::runtime)
    endif()
    if(TARGET Cellerator::architecture_provider)
        target_link_libraries(cellerator_compiler_smoke_accelerator PRIVATE Cellerator::architecture_provider)
    endif()
    set_target_properties(cellerator_compiler_smoke_accelerator PROPERTIES CUDA_ARCHITECTURES 70)
endfunction()
