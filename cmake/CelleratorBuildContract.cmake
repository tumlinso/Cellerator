include_guard(GLOBAL)

# Keep the same compiler and generator for the host and substrate builds.
if(NOT CMAKE_GENERATOR MATCHES "^Ninja")
    message(FATAL_ERROR "Cellerator requires Ninja 1.11 or newer")
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
if(NOT DEFINED CMAKE_CXX_SCAN_FOR_MODULES)
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CelleratorRetiredComponents.cmake")
