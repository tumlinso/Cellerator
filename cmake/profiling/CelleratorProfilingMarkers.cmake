option(CELLERATOR_ENABLE_PROFILING_MARKERS
       "Enable optional static cold profiling markers" OFF)

function(cellerator_configure_profiling_markers target)
    if(CELLERATOR_ENABLE_PROFILING_MARKERS)
        target_compile_definitions(${target} PRIVATE
            CELLERATOR_ENABLE_PROFILING_MARKERS=1)
    endif()
endfunction()
