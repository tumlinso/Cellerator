include(src/compute/CMakeLists.txt)
include(src/compute/current_targets.cmake)
include(src/execution/legacy_abi_targets.cmake)
include(src/preprocess/CMakeLists.txt)
include(src/interop/CMakeLists.txt)

if(CELLERATOR_ENABLE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE CELLERATOR_PYBIND11_CMAKE_DIR
            RESULT_VARIABLE CELLERATOR_PYBIND11_CMAKE_DIR_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(CELLERATOR_PYBIND11_CMAKE_DIR_STATUS EQUAL 0 AND NOT "${CELLERATOR_PYBIND11_CMAKE_DIR}" STREQUAL "")
            list(PREPEND CMAKE_PREFIX_PATH "${CELLERATOR_PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    if(NOT TARGET CellShard::export)
        message(FATAL_ERROR "CELLERATOR_ENABLE_PYTHON requires CellShard::export")
    endif()

    pybind11_add_module(_cellerator
        python/module.cc
    )
    target_include_directories(_cellerator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_cellerator PRIVATE
        Cellerator::preprocess
        CellShard::export
        CUDA::cudart
    )
    set_target_properties(_cellerator PROPERTIES
        OUTPUT_NAME _cellerator
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    cellerator_enable_perf(_cellerator)

    install(TARGETS _cellerator
        LIBRARY DESTINATION cellerator
        RUNTIME DESTINATION cellerator
    )
    install(FILES
        python/cellerator/__init__.py
        python/cellerator/pp.py
        DESTINATION cellerator
    )
endif()


include(examples/CMakeLists.txt)
include(tools/CMakeLists.txt)
