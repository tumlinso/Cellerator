include_guard(GLOBAL)

function(cellerator_configure_legacy_cuda_target target)
    if(NOT CELLERATOR_HAS_CUDA)
        message(FATAL_ERROR "legacy CUDA target requested in a host-only build")
    endif()
    find_package(CUDAToolkit REQUIRED)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(${target} PRIVATE CUDA::cudart)
endfunction()

function(cellerator_include_cuda_provider_manifest)
    if(CELLERATOR_HAS_CUDA)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/providers/CelleratorProviderPolicy.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/providers/CelleratorProviderTargets.cmake)
    endif()
endfunction()
