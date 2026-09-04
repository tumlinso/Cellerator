include_guard(GLOBAL)

function(cellerator_add_compiler_component target alias)
    if(NOT TARGET ${target})
        add_library(${target} INTERFACE)
        add_library(Cellerator::${alias} ALIAS ${target})
        target_include_directories(${target} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
    endif()
endfunction()

function(cellerator_create_compiler_component_targets)
    cellerator_add_compiler_component(cellerator_compiler_core CompilerCore)
    cellerator_add_compiler_component(cellerator_compiler_diagnostics CompilerDiagnostics)
    cellerator_add_compiler_component(cellerator_compiler_ceir CEIR)
    cellerator_add_compiler_component(cellerator_compiler_profiles CompilerProfiles)
    cellerator_add_compiler_component(cellerator_compiler_frontend CompilerFrontend)
    cellerator_add_compiler_component(cellerator_compiler_planning CompilerPlanning)
    cellerator_add_compiler_component(cellerator_compiler_realization CompilerRealization)
    cellerator_add_compiler_component(cellerator_compiler_backends CompilerBackends)
    cellerator_add_compiler_component(cellerator_compiler_tooling CompilerTooling)

    target_link_libraries(cellerator_compiler_diagnostics INTERFACE Cellerator::CompilerCore)
    target_link_libraries(cellerator_compiler_ceir INTERFACE Cellerator::CompilerCore Cellerator::CompilerDiagnostics)
    target_link_libraries(cellerator_compiler_profiles INTERFACE Cellerator::CEIR)
    target_link_libraries(cellerator_compiler_frontend INTERFACE Cellerator::CEIR Cellerator::CompilerDiagnostics)
    target_link_libraries(cellerator_compiler_planning INTERFACE Cellerator::CEIR Cellerator::CompilerProfiles)
    target_link_libraries(cellerator_compiler_realization INTERFACE Cellerator::CompilerPlanning)
    target_link_libraries(cellerator_compiler_backends INTERFACE Cellerator::CompilerRealization)
    target_link_libraries(cellerator_compiler_tooling INTERFACE
        Cellerator::CompilerFrontend Cellerator::CompilerBackends)
endfunction()
