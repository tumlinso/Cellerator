include_guard(GLOBAL)

function(cellerator_remove_provider_sources target output_var)
    get_target_property(provider_host_sources ${target} SOURCES)
    if(NOT provider_host_sources)
        set(${output_var} "" PARENT_SCOPE)
        return()
    endif()

    set(provider_remaining_sources)
    foreach(provider_source IN LISTS provider_host_sources)
        if(provider_source IN_LIST CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES
           OR provider_source IN_LIST CELLERATOR_NVIDIA_SM70_PROVIDER_SOURCES)
            continue()
        endif()
        list(APPEND provider_remaining_sources "${provider_source}")
    endforeach()
    set(${output_var} "${provider_remaining_sources}" PARENT_SCOPE)
endfunction()

function(cellerator_configure_provider_target target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${target} PUBLIC
        Cellerator::operation_core
        Cellerator::runtime
        Cellerator::semantic_geometry_v1
        CellPack::execution_image_v2
        CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES)
endfunction()

function(cellerator_finalize_cuda_provider_targets)
    if(NOT TARGET cellerator_architecture_provider)
        message(FATAL_ERROR
            "provider target split requires cellerator_architecture_provider")
    endif()

    cellerator_remove_provider_sources(cellerator_architecture_provider
        provider_architecture_sources)
    set_property(TARGET cellerator_architecture_provider PROPERTY SOURCES
        "${provider_architecture_sources}")

    add_library(cellerator_provider_common STATIC
        ${CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES})
    add_library(Cellerator::provider_common ALIAS
        cellerator_provider_common)
    cellerator_configure_provider_target(cellerator_provider_common)
    cellerator_apply_provider_build_policy(cellerator_provider_common)

    # The generic target is an explicit target-neutral provider surface.  It
    # owns no accelerator-specific source and therefore remains an interface
    # target until a generic source-linked provider is added to the catalog.
    add_library(cellerator_provider_generic INTERFACE)
    add_library(Cellerator::provider_generic ALIAS
        cellerator_provider_generic)
    target_link_libraries(cellerator_provider_generic INTERFACE
        Cellerator::provider_common)

    if(CELLERATOR_PROVIDER_INCLUDE_SM70)
        add_library(cellerator_provider_sm70 STATIC
            ${CELLERATOR_NVIDIA_SM70_PROVIDER_SOURCES})
        cellerator_configure_provider_target(cellerator_provider_sm70)
        cellerator_apply_provider_build_policy(cellerator_provider_sm70)
        set_property(TARGET cellerator_provider_sm70 PROPERTY
            CUDA_ARCHITECTURES 70)
        target_link_libraries(cellerator_architecture_provider PUBLIC
            cellerator_provider_sm70)
        set(CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS
            "#include <Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh>")
        set(CELLERATOR_PROVIDER_MANIFEST_ENTRIES
            "        &providers::nvidia::register_sm70_provider_v1,")
    else()
        add_library(cellerator_provider_sm70 INTERFACE)
        set(CELLERATOR_PROVIDER_MANIFEST_DECLARATIONS "")
        set(CELLERATOR_PROVIDER_MANIFEST_ENTRIES "")
    endif()
    add_library(Cellerator::provider_sm70 ALIAS cellerator_provider_sm70)

    target_link_libraries(cellerator_architecture_provider PUBLIC
        cellerator_provider_common)
    if(CELLERATOR_PROVIDER_INCLUDE_GENERIC)
        target_link_libraries(cellerator_architecture_provider PUBLIC
            cellerator_provider_generic)
    endif()

    set(provider_manifest_directory
        "${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/compute/architecture")
    file(MAKE_DIRECTORY "${provider_manifest_directory}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/provider_manifest/compiled_provider_manifest.hh.in"
        "${provider_manifest_directory}/compiled_provider_manifest.hh"
        @ONLY)

    add_library(cellerator_provider_manifest INTERFACE)
    add_library(Cellerator::provider_manifest ALIAS
        cellerator_provider_manifest)
    target_include_directories(cellerator_provider_manifest INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(cellerator_provider_manifest INTERFACE
        Cellerator::architecture_provider)
endfunction()
