include_guard(GLOBAL)

include(CMakeParseArguments)

# Declare a source-linked CUDA provider without assigning any optimization
# policy to it. Inclusion, target aliases, architecture code generation, and
# tuning are deliberately separate choices.
function(cellerator_add_cuda_provider)
    set(options)
    set(one_value_args
        NAME
        TARGET
        OPTION
        DEFAULT
        REGISTRATION
        DECLARATION
    )
    set(multi_value_args
        SOURCES
        LINK_LIBRARIES
        INCLUDE_DIRECTORIES
        ARCHITECTURES
        ALIASES
    )
    cmake_parse_arguments(PARSE_ARGV 0 provider
        "${options}" "${one_value_args}" "${multi_value_args}")

    if(provider_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "cellerator_add_cuda_provider received unknown arguments: "
            "${provider_UNPARSED_ARGUMENTS}")
    endif()
    foreach(required NAME TARGET OPTION REGISTRATION DECLARATION)
        if(NOT provider_${required})
            message(FATAL_ERROR
                "cellerator_add_cuda_provider requires ${required}")
        endif()
    endforeach()
    if(NOT provider_SOURCES)
        message(FATAL_ERROR
            "cellerator_add_cuda_provider requires at least one source")
    endif()

    if(provider_DEFAULT STREQUAL "")
        set(provider_DEFAULT OFF)
    endif()
    option(${provider_OPTION}
        "Build the source-linked ${provider_NAME} CUDA provider"
        ${provider_DEFAULT})
    if(NOT ${provider_OPTION})
        return()
    endif()

    if(TARGET ${provider_TARGET})
        message(FATAL_ERROR
            "CUDA provider target already exists: ${provider_TARGET}")
    endif()
    add_library(${provider_TARGET} STATIC ${provider_SOURCES})
    target_compile_features(${provider_TARGET} PUBLIC cxx_std_17)
    set_target_properties(${provider_TARGET} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
    if(provider_INCLUDE_DIRECTORIES)
        target_include_directories(${provider_TARGET} PUBLIC
            ${provider_INCLUDE_DIRECTORIES})
    endif()
    if(provider_LINK_LIBRARIES)
        target_link_libraries(${provider_TARGET} PUBLIC
            ${provider_LINK_LIBRARIES})
    endif()
    if(provider_ARCHITECTURES)
        set_property(TARGET ${provider_TARGET} PROPERTY CUDA_ARCHITECTURES
            "${provider_ARCHITECTURES}")
    endif()
    foreach(alias IN LISTS provider_ALIASES)
        if(TARGET ${alias})
            message(FATAL_ERROR "CUDA provider alias already exists: ${alias}")
        endif()
        add_library(${alias} ALIAS ${provider_TARGET})
    endforeach()

    get_property(provider_declarations GLOBAL PROPERTY
        CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS)
    get_property(provider_entries GLOBAL PROPERTY
        CELLERATOR_PROVIDER_MANIFEST_ENTRIES)
    string(APPEND provider_declarations "${provider_DECLARATION}\n")
    string(APPEND provider_entries "        &${provider_REGISTRATION},\n")
    set_property(GLOBAL PROPERTY CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS
        "${provider_declarations}")
    set_property(GLOBAL PROPERTY CELLERATOR_PROVIDER_MANIFEST_ENTRIES
        "${provider_entries}")
endfunction()

# Apply only tuning flags named by the caller. This helper intentionally has no
# default fast-math, cache, register-count, launch-bound, or host-ISA policy.
function(cellerator_apply_cuda_provider_tuning target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "unknown CUDA provider target: ${target}")
    endif()
    set(options)
    set(one_value_args)
    set(multi_value_args CUDA_FLAGS CXX_FLAGS LINK_LIBRARIES)
    cmake_parse_arguments(PARSE_ARGV 1 tuning
        "${options}" "${one_value_args}" "${multi_value_args}")
    if(tuning_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "cellerator_apply_cuda_provider_tuning received unknown arguments: "
            "${tuning_UNPARSED_ARGUMENTS}")
    endif()
    if(tuning_CUDA_FLAGS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${tuning_CUDA_FLAGS}>)
    endif()
    if(tuning_CXX_FLAGS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${tuning_CXX_FLAGS}>)
    endif()
    if(tuning_LINK_LIBRARIES)
        target_link_libraries(${target} PRIVATE ${tuning_LINK_LIBRARIES})
    endif()
endfunction()

function(cellerator_generate_cuda_provider_manifest output_path)
    if(NOT IS_ABSOLUTE "${output_path}")
        set(output_path "${CMAKE_CURRENT_BINARY_DIR}/${output_path}")
    endif()
    get_filename_component(output_directory "${output_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    get_property(CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS GLOBAL PROPERTY
        CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS)
    get_property(CELLERATOR_PROVIDER_MANIFEST_ENTRIES GLOBAL PROPERTY
        CELLERATOR_PROVIDER_MANIFEST_ENTRIES)
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cellerator_provider_manifest.hh.in"
        "${output_path}"
        @ONLY
    )
endfunction()
