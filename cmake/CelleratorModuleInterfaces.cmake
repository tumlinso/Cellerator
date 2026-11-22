include_guard(GLOBAL)

function(cellerator_require_module_scanner)
    string(REGEX MATCH "^[0-9]+" clang_major "${CMAKE_CXX_COMPILER_VERSION}")
    if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
        find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
            NAMES clang-scan-deps-${clang_major} clang-scan-deps REQUIRED)
    endif()
    execute_process(COMMAND "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}" --version
        RESULT_VARIABLE scan_status OUTPUT_VARIABLE scan_version ERROR_QUIET)
    if(NOT scan_status EQUAL 0 OR NOT scan_version MATCHES "LLVM version ${clang_major}\\.")
        message(FATAL_ERROR
            "Cellerator named modules require clang-scan-deps matching Clang ${clang_major}. "
            "Set CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS to the matching executable.")
    endif()
    set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}" PARENT_SCOPE)
endfunction()

# Rebuild each installed interface once in the consumer's compiler context.
# CMake 3.28 otherwise emits a duplicate synthetic target when multiple
# consumers link the same imported CXX_MODULES file set.
function(cellerator_add_installed_module_interface target binary)
    get_target_property(_module_files ${binary} CXX_MODULE_SET)
    get_target_property(_module_dirs ${binary} CXX_MODULE_DIRS)
    if(NOT _module_files OR NOT _module_dirs)
        message(FATAL_ERROR "Cellerator module target ${binary} has no installed interface sources")
    endif()
    add_library(${target} STATIC EXCLUDE_FROM_ALL)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
    target_sources(${target} PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS ${_module_dirs} FILES ${_module_files})
    target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()

function(cellerator_attach_module target binary)
    string(REPLACE "Cellerator::" "cellerator_" provider "${target}_installed_interface")
    if(NOT TARGET ${provider})
        cellerator_add_installed_module_interface(${provider} ${binary})
        foreach(property IN ITEMS INCLUDE_DIRECTORIES COMPILE_OPTIONS COMPILE_DEFINITIONS LINK_LIBRARIES)
            get_target_property(usage ${binary} INTERFACE_${property})
            if(usage)
                set_property(TARGET ${provider} PROPERTY ${property} "${usage}")
            endif()
        endforeach()
    endif()
    get_target_property(links ${target} INTERFACE_LINK_LIBRARIES)
    if(NOT provider IN_LIST links)
        target_link_libraries(${target} INTERFACE ${provider})
    endif()
endfunction()

function(cellerator_import_module name binary)
    if(TARGET Cellerator::${name})
        return()
    endif()
    set(_target cellerator_${name}_interfaces)
    cellerator_add_installed_module_interface(${_target} ${binary})
    target_include_directories(${_target} PUBLIC
        "$<TARGET_PROPERTY:${binary},INTERFACE_INCLUDE_DIRECTORIES>")
    if(ARGN)
        target_link_libraries(${_target} PUBLIC ${ARGN})
    endif()
    add_library(Cellerator::${name} INTERFACE IMPORTED GLOBAL)
    set_property(TARGET Cellerator::${name} PROPERTY INTERFACE_LINK_LIBRARIES ${_target})
endfunction()
