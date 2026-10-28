include_guard(GLOBAL)

# Compiler-only configuration starts from a CXX root project. Accelerator
# languages are enabled later, after an explicit backend policy decision.
set(CELLERATOR_COMPILER_ROOT_LANGUAGES "CXX")
set(CELLERATOR_COMPILER_ACCELERATOR_ENABLEMENT "OPTIONAL")
set(CELLERATOR_COMPILER_DEFAULT_ACCELERATOR_POLICY "AUTO")

function(cellerator_validate_host_only_root_project_contract)
    if(NOT CELLERATOR_COMPILER_ROOT_LANGUAGES STREQUAL "CXX")
        message(FATAL_ERROR
            "The Cellerator compiler root project must require only CXX")
    endif()
    if(CMAKE_CUDA_COMPILER_LOADED)
        message(FATAL_ERROR
            "Host-only compiler configuration must not load the CUDA language")
    endif()
endfunction()
