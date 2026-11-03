include_guard(GLOBAL)

# Rebuild each installed interface once in the consumer's compiler context.
# CMake 3.28 otherwise emits a duplicate synthetic target when multiple
# consumers link the same imported CXX_MODULES file set.
function(cellerator_import_module name binary)
    if(TARGET Cellerator::${name})
        return()
    endif()
    get_target_property(_module_files ${binary} CXX_MODULE_SET)
    get_target_property(_module_dirs ${binary} CXX_MODULE_DIRS)
    if(NOT _module_files OR NOT _module_dirs)
        message(FATAL_ERROR "Cellerator module ${name} has no installed interface sources")
    endif()
    set(_target cellerator_${name}_interfaces)
    add_library(${_target} STATIC EXCLUDE_FROM_ALL)
    set_target_properties(${_target} PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
    target_sources(${_target} PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS ${_module_dirs} FILES ${_module_files})
    target_compile_features(${_target} PUBLIC cxx_std_20)
    target_include_directories(${_target} PUBLIC
        "$<TARGET_PROPERTY:${binary},INTERFACE_INCLUDE_DIRECTORIES>")
    if(ARGN)
        target_link_libraries(${_target} PUBLIC ${ARGN})
    endif()
    add_library(Cellerator::${name} INTERFACE IMPORTED GLOBAL)
    set_property(TARGET Cellerator::${name} PROPERTY INTERFACE_LINK_LIBRARIES ${_target})
endfunction()
