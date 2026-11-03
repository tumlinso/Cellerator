include_guard(GLOBAL)

# The native AnnData dependency and Cellerator's public modules use CMake's
# scanned C++20 module graph. Keep the same compiler and generator for every
# build, including the host-only and substrate projections.
if(NOT CMAKE_GENERATOR MATCHES "^Ninja")
    message(FATAL_ERROR "Cellerator requires Ninja 1.11 or newer for C++20 module scanning")
endif()
execute_process(COMMAND "${CMAKE_MAKE_PROGRAM}" --version
    RESULT_VARIABLE _cellerator_ninja_status
    OUTPUT_VARIABLE _cellerator_ninja_version OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _cellerator_ninja_status EQUAL 0 OR _cellerator_ninja_version VERSION_LESS 1.11)
    message(FATAL_ERROR "Cellerator requires Ninja 1.11 or newer")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
    message(FATAL_ERROR "Cellerator requires Clang 18 or newer; configure with -DCMAKE_CXX_COMPILER=clang++-18")
endif()
string(REGEX MATCH "^[0-9]+" _cellerator_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
        NAMES clang-scan-deps-${_cellerator_clang_major} clang-scan-deps REQUIRED)
endif()
execute_process(COMMAND "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}" --version
    RESULT_VARIABLE _cellerator_scan_status
    OUTPUT_VARIABLE _cellerator_scan_version ERROR_VARIABLE _cellerator_scan_error)
if(NOT _cellerator_scan_status EQUAL 0 OR
   NOT _cellerator_scan_version MATCHES "LLVM version ${_cellerator_clang_major}\\.")
    message(FATAL_ERROR
        "Cellerator requires clang-scan-deps matching Clang ${_cellerator_clang_major}; "
        "set CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS to the matching executable")
endif()

foreach(_cellerator_language IN ITEMS CXX CUDA)
    if(DEFINED CMAKE_${_cellerator_language}_STANDARD AND
       CMAKE_${_cellerator_language}_STANDARD LESS 20)
        message(FATAL_ERROR "Cellerator requires CMAKE_${_cellerator_language}_STANDARD >= 20")
    endif()
    if(NOT DEFINED CMAKE_${_cellerator_language}_STANDARD)
        set(CMAKE_${_cellerator_language}_STANDARD 20)
    endif()
    set(CMAKE_${_cellerator_language}_STANDARD_REQUIRED ON)
    set(CMAKE_${_cellerator_language}_EXTENSIONS OFF)
endforeach()
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

find_package(anndata CONFIG REQUIRED)
if(NOT ANNDATA_HAS_HDF5)
    message(FATAL_ERROR "Cellerator requires an anndata package built with ANNDATA_ENABLE_HDF5=ON")
endif()
if(ANNDATA_HAS_ZARR)
    message(FATAL_ERROR "Cellerator requires an anndata package built with ANNDATA_ENABLE_ZARR=OFF")
endif()
