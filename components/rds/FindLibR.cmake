find_program(CELLERATOR_R_EXECUTABLE NAMES R)
if(CELLERATOR_R_EXECUTABLE)
    execute_process(COMMAND "${CELLERATOR_R_EXECUTABLE}" RHOME
        RESULT_VARIABLE _libr_home_status OUTPUT_VARIABLE _libr_home
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_libr_home_status STREQUAL "0" AND IS_DIRECTORY "${_libr_home}")
        execute_process(COMMAND "${CELLERATOR_R_EXECUTABLE}" CMD config --cppflags
            RESULT_VARIABLE _libr_flags_status OUTPUT_VARIABLE _libr_flags
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(_libr_include_hints "${_libr_home}/include")
        if(_libr_flags_status STREQUAL "0")
            separate_arguments(_libr_flags UNIX_COMMAND "${_libr_flags}")
            foreach(_flag IN LISTS _libr_flags)
                if(_flag MATCHES "^-I(.+)$")
                    list(PREPEND _libr_include_hints "${CMAKE_MATCH_1}")
                endif()
            endforeach()
        endif()
        find_path(LibR_INCLUDE_DIR Rinternals.h PATHS ${_libr_include_hints}
            NO_DEFAULT_PATH NO_CACHE)
        find_library(LibR_LIBRARY NAMES "${CMAKE_SHARED_LIBRARY_PREFIX}R${CMAKE_SHARED_LIBRARY_SUFFIX}"
            PATHS "${_libr_home}/lib" NO_DEFAULT_PATH NO_CACHE)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibR REQUIRED_VARS LibR_INCLUDE_DIR LibR_LIBRARY
    REASON_FAILURE_MESSAGE "Use an R installation with development headers and shared libR. Set CELLERATOR_R_EXECUTABLE to its R executable if needed.")
if(LibR_FOUND AND NOT TARGET LibR::LibR)
    add_library(LibR::LibR SHARED IMPORTED)
    set_target_properties(LibR::LibR PROPERTIES
        IMPORTED_LOCATION "${LibR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LibR_INCLUDE_DIR}")
endif()
