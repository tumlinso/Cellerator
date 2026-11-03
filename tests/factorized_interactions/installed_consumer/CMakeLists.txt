cmake_minimum_required(VERSION 3.20)
project(CelleratorFactorizedInteractionInstalledConsumer LANGUAGES CXX)

find_package(Cellerator CONFIG REQUIRED
    COMPONENTS factorized_interaction_cuda
    OPTIONAL_COMPONENTS nn)
if(NOT Cellerator_factorized_interaction_cuda_FOUND)
    message(FATAL_ERROR "Installed factorized-interaction component was not reported found")
endif()
if(Cellerator_nn_FOUND)
    message(FATAL_ERROR "Component-only install incorrectly reported optional NN as available")
endif()
get_target_property(_memory_archive_first Cellerator::memory_cuda
    IMPORTED_LOCATION_RELEASE)
# Repeating the same request is an idempotent load of the same complete set of
# imported targets; a partial subset must not trip CMake's export-set guard.
find_package(Cellerator CONFIG REQUIRED
    COMPONENTS factorized_interaction_cuda)
get_target_property(_memory_archive_second Cellerator::memory_cuda
    IMPORTED_LOCATION_RELEASE)
if(NOT "${_memory_archive_first}" STREQUAL "${_memory_archive_second}")
    message(FATAL_ERROR "Repeated component loading changed the memory target identity")
endif()

foreach(_target IN ITEMS
        Cellerator::support
        Cellerator::layout
        Cellerator::memory_cuda
        Cellerator::factorized_interaction_cuda)
    if(NOT TARGET ${_target})
        message(FATAL_ERROR "Installed component did not define ${_target}")
    endif()
endforeach()
get_target_property(_memory_archive Cellerator::memory_cuda
    IMPORTED_LOCATION_RELEASE)
if(NOT _memory_archive OR NOT EXISTS "${_memory_archive}")
    message(FATAL_ERROR "Canonical Cellerator::memory_cuda archive is unavailable")
endif()

add_executable(factorized_interaction_installed_consumer main.cc)
target_compile_features(factorized_interaction_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(factorized_interaction_installed_consumer PRIVATE
    Cellerator::factorized_interaction_cuda)
