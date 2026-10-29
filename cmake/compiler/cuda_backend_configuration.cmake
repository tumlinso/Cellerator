include_guard(GLOBAL)

set(CELLERATOR_ENABLE_CUDA "AUTO" CACHE STRING
    "Enable CUDA compiler backends: AUTO, ON, or OFF")
set_property(CACHE CELLERATOR_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)

macro(cellerator_configure_optional_cuda_backend)
    string(TOUPPER "${CELLERATOR_ENABLE_CUDA}" cellerator_cuda_mode)
    if(NOT cellerator_cuda_mode MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "CELLERATOR_ENABLE_CUDA must be AUTO, ON, or OFF")
    endif()

    if(cellerator_cuda_mode STREQUAL "OFF")
        set(CELLERATOR_HAS_CUDA FALSE)
        return()
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CELLERATOR_HAS_CUDA TRUE)
    elseif(cellerator_cuda_mode STREQUAL "ON")
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_CUDA=ON requires a usable CUDA compiler")
    else()
        set(CELLERATOR_HAS_CUDA FALSE)
    endif()
endmacro()
