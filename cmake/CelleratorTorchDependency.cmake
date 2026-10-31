# Shared Torch discovery for optional ATen-backed Cellerator components.
# Keep the Python binding and native C++ surfaces on one Torch/ATen ABI when
# they are built together, while allowing standalone native builds to point at
# an explicit LibTorch package.
include_guard(GLOBAL)

include(CMakeFindDependencyMacro)

set(CELLERATOR_LIBTORCH_ROOT "" CACHE PATH
    "Optional LibTorch package prefix for ATen-backed Cellerator components")

# Prefer the Torch package belonging to the selected Python interpreter when
# available, so pybind tensor casters and native ATen code share the same ABI.
if(CELLERATOR_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE cellerator_torch_python_status
        OUTPUT_VARIABLE cellerator_torch_python_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(cellerator_torch_python_status EQUAL 0 AND cellerator_torch_python_prefix)
        list(PREPEND CMAKE_PREFIX_PATH "${cellerator_torch_python_prefix}")
        set(cellerator_torch_python_dir "${cellerator_torch_python_prefix}/Torch")
        if(Torch_DIR)
            get_filename_component(cellerator_torch_requested_dir
                "${Torch_DIR}" REALPATH)
            get_filename_component(cellerator_torch_expected_dir
                "${cellerator_torch_python_dir}" REALPATH)
            if(NOT cellerator_torch_requested_dir STREQUAL cellerator_torch_expected_dir)
                message(FATAL_ERROR
                    "CELLERATOR_ENABLE_PYTHON requires Torch_DIR to match the selected Python's Torch package. "
                    "Python Torch_DIR is '${cellerator_torch_expected_dir}', "
                    "but Torch_DIR is '${cellerator_torch_requested_dir}'.")
            endif()
        endif()
        set(Torch_DIR "${cellerator_torch_python_dir}" CACHE PATH
            "Torch package belonging to the selected Python interpreter" FORCE)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
            RESULT_VARIABLE cellerator_torch_python_root_status
            OUTPUT_VARIABLE cellerator_torch_python_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(cellerator_torch_python_root_status EQUAL 0 AND cellerator_torch_python_root)
            list(PREPEND CMAKE_PREFIX_PATH "${cellerator_torch_python_root}")
        endif()
    endif()
endif()

if(CELLERATOR_LIBTORCH_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${CELLERATOR_LIBTORCH_ROOT}")
endif()

# TorchConfig loads Caffe2 as an implementation dependency. Keep it from the
# same selected installation instead of mixing a stale cached Caffe2 package.
if(Torch_DIR)
    get_filename_component(cellerator_torch_cmake_dir "${Torch_DIR}" REALPATH)
    get_filename_component(cellerator_torch_caffe2_dir
        "${cellerator_torch_cmake_dir}/../Caffe2" REALPATH)
    if(EXISTS "${cellerator_torch_caffe2_dir}/Caffe2Config.cmake")
        set(Caffe2_DIR "${cellerator_torch_caffe2_dir}" CACHE PATH
            "Caffe2 package belonging to the selected Torch installation" FORCE)
    endif()
endif()

# TorchConfig uses generic cache names for libraries. Resolve those only from
# the selected package to prevent stale entries from mixing installations.
foreach(cellerator_torch_cache_var IN ITEMS
        TORCH_LIBRARY c10_LIBRARY torch_cpu_LIBRARY torch_cuda_LIBRARY
        c10_cuda_LIBRARY kineto_LIBRARY C10_CUDA_LIBRARY)
    unset(${cellerator_torch_cache_var} CACHE)
endforeach()

if(CELLERATOR_ENABLE_PYTHON AND cellerator_torch_python_prefix)
    get_filename_component(cellerator_torch_selected_root
        "${cellerator_torch_python_prefix}/../.." REALPATH)
elseif(Torch_DIR)
    get_filename_component(cellerator_torch_selected_root
        "${Torch_DIR}/../../.." REALPATH)
elseif(CELLERATOR_LIBTORCH_ROOT)
    get_filename_component(cellerator_torch_selected_root
        "${CELLERATOR_LIBTORCH_ROOT}" REALPATH)
endif()
if(cellerator_torch_selected_root)
    foreach(cellerator_torch_library IN ITEMS
            "TORCH_LIBRARY;torch" "c10_LIBRARY;c10"
            "torch_cpu_LIBRARY;torch_cpu" "torch_cuda_LIBRARY;torch_cuda"
            "c10_cuda_LIBRARY;c10_cuda" "kineto_LIBRARY;kineto"
            "C10_CUDA_LIBRARY;c10_cuda")
        list(GET cellerator_torch_library 0 cellerator_torch_variable)
        list(GET cellerator_torch_library 1 cellerator_torch_name)
        find_library(${cellerator_torch_variable}
            NAMES ${cellerator_torch_name}
            PATHS "${cellerator_torch_selected_root}/lib"
            NO_DEFAULT_PATH)
    endforeach()
endif()

# Caffe2 consumes TORCH_CUDA_ARCH_LIST and can append global NVCC flags. Keep
# it aligned with explicit CMake architectures, without probing a device.
set(cellerator_torch_saved_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_COMPILER_LOADED
   AND CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(all|native)$"
   AND (NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST}
        OR "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
   AND (NOT DEFINED TORCH_CUDA_ARCH_LIST
        OR "${TORCH_CUDA_ARCH_LIST}" STREQUAL ""))
    set(cellerator_torch_arch_entries)
    foreach(cellerator_torch_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        set(cellerator_torch_cuda_arch_digits "${cellerator_torch_cuda_arch}")
        if(cellerator_torch_cuda_arch_digits MATCHES "^([0-9])([0-9])(-real|-virtual)?$")
            list(APPEND cellerator_torch_arch_entries
                "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
        else()
            set(cellerator_torch_arch_entries)
            break()
        endif()
    endforeach()
    if(cellerator_torch_arch_entries)
        string(REPLACE ";" " " cellerator_torch_arch_list
            "${cellerator_torch_arch_entries}")
        set(TORCH_CUDA_ARCH_LIST "${cellerator_torch_arch_list}" CACHE STRING
            "CUDA architectures passed to the selected Torch package")
        set(ENV{TORCH_CUDA_ARCH_LIST} "${cellerator_torch_arch_list}")
    endif()
endif()

find_package(Torch REQUIRED)
if(cellerator_torch_saved_cuda_architectures)
    set(CMAKE_CUDA_ARCHITECTURES "${cellerator_torch_saved_cuda_architectures}" CACHE STRING
        "CUDA architectures selected for Cellerator" FORCE)
endif()

if(TARGET torch_library)
    set(CELLERATOR_TORCH_LINK_TARGETS torch torch_library)
elseif(TARGET torch)
    set(CELLERATOR_TORCH_LINK_TARGETS torch)
else()
    set(CELLERATOR_TORCH_LINK_TARGETS ${TORCH_LIBRARIES})
endif()

if(CELLERATOR_ENABLE_PYTHON AND cellerator_torch_python_prefix)
    set(CELLERATOR_TORCH_LIB_DIR "${cellerator_torch_python_prefix}/../../lib")
    get_filename_component(CELLERATOR_TORCH_LIB_DIR
        "${CELLERATOR_TORCH_LIB_DIR}" REALPATH)
elseif(CELLERATOR_LIBTORCH_ROOT)
    set(CELLERATOR_TORCH_LIB_DIR "${CELLERATOR_LIBTORCH_ROOT}/lib")
elseif(TORCH_INSTALL_PREFIX)
    set(CELLERATOR_TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(TORCH_CXX_FLAGS)
    separate_arguments(CELLERATOR_TORCH_CXX_OPTIONS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

set(CELLERATOR_TORCH_DEPENDENCY_READY TRUE)
