include_guard(GLOBAL)

function(cellerator_validate_frozen_compiler_target_graph)
    foreach(target IN ITEMS cellerator_compiler_core cellerator_compiler_diagnostics
            cellerator_compiler_ceir cellerator_compiler_profiles
            cellerator_compiler_frontend cellerator_compiler_planning
            cellerator_compiler_realization cellerator_compiler_backends
            cellerator_compiler_tooling)
        if(NOT TARGET ${target})
            message(FATAL_ERROR "missing compiler thin-waist target: ${target}")
        endif()
    endforeach()
    foreach(target IN ITEMS cellerator_compiler_core cellerator_compiler_diagnostics
            cellerator_compiler_ceir cellerator_compiler_profiles
            cellerator_compiler_frontend cellerator_compiler_planning
            cellerator_compiler_tooling)
        get_target_property(links ${target} INTERFACE_LINK_LIBRARIES)
        if(links MATCHES "CUDA::")
            message(FATAL_ERROR "host compiler target has unconditional CUDA edge: ${target}")
        endif()
    endforeach()
endfunction()
