include(GNUInstallDirs)
include(CMakeFindDependencyMacro)

if(NOT CELLERATOR_ENABLE_TOKENIZATION)
    return()
endif()

set(CELLERATOR_LIBTORCH_ROOT "" CACHE PATH
    "Optional libtorch package prefix for the ATen tokenization component")

# Prefer the Torch package belonging to the selected Python interpreter when
# available, so the pybind tensor caster and libtorch ABI stay in agreement.
# C++-only consumers may instead set Torch_DIR, CMAKE_PREFIX_PATH, or the
# explicit CELLERATOR_LIBTORCH_ROOT.
if(CELLERATOR_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE cellerator_tokenization_python_torch_status
        OUTPUT_VARIABLE cellerator_tokenization_python_torch_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(cellerator_tokenization_python_torch_status EQUAL 0
       AND cellerator_tokenization_python_torch_prefix)
        list(PREPEND CMAKE_PREFIX_PATH
            "${cellerator_tokenization_python_torch_prefix}")
        set(cellerator_tokenization_python_torch_dir
            "${cellerator_tokenization_python_torch_prefix}/Torch")
        if(Torch_DIR)
            get_filename_component(cellerator_tokenization_requested_torch_dir
                "${Torch_DIR}" REALPATH)
            get_filename_component(cellerator_tokenization_expected_torch_dir
                "${cellerator_tokenization_python_torch_dir}" REALPATH)
            if(NOT cellerator_tokenization_requested_torch_dir
               STREQUAL cellerator_tokenization_expected_torch_dir)
                message(FATAL_ERROR
                    "CELLERATOR_ENABLE_PYTHON requires Torch_DIR to match the selected Python's Torch package. "
                    "Python Torch_DIR is '${cellerator_tokenization_expected_torch_dir}', "
                    "but Torch_DIR is '${cellerator_tokenization_requested_torch_dir}'.")
            endif()
        endif()
        set(Torch_DIR "${cellerator_tokenization_python_torch_dir}" CACHE PATH
            "Torch package belonging to the selected Python interpreter" FORCE)
    endif()
endif()

if(CELLERATOR_LIBTORCH_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${CELLERATOR_LIBTORCH_ROOT}")
endif()
# TorchConfig loads Caffe2 as an implementation dependency. Force the sibling
# package from the same selected installation; otherwise an unrelated cached
# Caffe2_DIR can mix CPU headers with a different Torch binary.
if(Torch_DIR)
    get_filename_component(cellerator_tokenization_torch_cmake_dir
        "${Torch_DIR}" REALPATH)
    get_filename_component(cellerator_tokenization_package_cmake_dir
        "${cellerator_tokenization_torch_cmake_dir}/../Caffe2" REALPATH)
    if(EXISTS "${cellerator_tokenization_package_cmake_dir}/Caffe2Config.cmake")
        set(Caffe2_DIR "${cellerator_tokenization_package_cmake_dir}" CACHE PATH
            "Caffe2 package belonging to the selected Torch installation" FORCE)
    endif()
endif()
# TorchConfig uses find_library results cached under generic names such as
# TORCH_LIBRARY. Resolve those names only inside the package selected by
# Torch_DIR, so a stale cache or a system-wide default cannot mix installations.
foreach(cellerator_tokenization_torch_cache_var IN ITEMS
        TORCH_LIBRARY c10_LIBRARY torch_cpu_LIBRARY torch_cuda_LIBRARY
        c10_cuda_LIBRARY kineto_LIBRARY C10_CUDA_LIBRARY)
    unset(${cellerator_tokenization_torch_cache_var} CACHE)
endforeach()
if(CELLERATOR_ENABLE_PYTHON AND cellerator_tokenization_python_torch_prefix)
    get_filename_component(cellerator_tokenization_selected_torch_root
        "${cellerator_tokenization_python_torch_prefix}/../.." REALPATH)
elseif(Torch_DIR)
    get_filename_component(cellerator_tokenization_selected_torch_root
        "${Torch_DIR}/../../.." REALPATH)
elseif(CELLERATOR_LIBTORCH_ROOT)
    get_filename_component(cellerator_tokenization_selected_torch_root
        "${CELLERATOR_LIBTORCH_ROOT}" REALPATH)
endif()
if(cellerator_tokenization_selected_torch_root)
    foreach(cellerator_tokenization_torch_library IN ITEMS
            "TORCH_LIBRARY;torch" "c10_LIBRARY;c10"
            "torch_cpu_LIBRARY;torch_cpu" "torch_cuda_LIBRARY;torch_cuda"
            "c10_cuda_LIBRARY;c10_cuda" "kineto_LIBRARY;kineto"
            "C10_CUDA_LIBRARY;c10_cuda")
        list(GET cellerator_tokenization_torch_library 0
            cellerator_tokenization_torch_variable)
        list(GET cellerator_tokenization_torch_library 1
            cellerator_tokenization_torch_name)
        find_library(${cellerator_tokenization_torch_variable}
            NAMES ${cellerator_tokenization_torch_name}
            PATHS "${cellerator_tokenization_selected_torch_root}/lib"
            NO_DEFAULT_PATH)
    endforeach()
endif()
# Caffe2's CUDA configuration consumes TORCH_CUDA_ARCH_LIST and may append
# global NVCC flags. Keep it aligned with the explicitly selected CMake
# architectures, without probing a device or overriding a caller's choice.
set(cellerator_tokenization_saved_cuda_architectures
    "${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_COMPILER_LOADED
   AND CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(all|native)$"
   AND (NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST}
        OR "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
   AND (NOT DEFINED TORCH_CUDA_ARCH_LIST
        OR "${TORCH_CUDA_ARCH_LIST}" STREQUAL ""))
    set(cellerator_tokenization_torch_arch_entries)
    foreach(cellerator_tokenization_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        set(cellerator_tokenization_cuda_arch_digits
            "${cellerator_tokenization_cuda_arch}")
        if(cellerator_tokenization_cuda_arch_digits MATCHES "^([0-9])([0-9])(-real|-virtual)?$")
            list(APPEND cellerator_tokenization_torch_arch_entries
                "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
        else()
            set(cellerator_tokenization_torch_arch_entries)
            break()
        endif()
    endforeach()
    if(cellerator_tokenization_torch_arch_entries)
        string(REPLACE ";" " " cellerator_tokenization_torch_arch_list
            "${cellerator_tokenization_torch_arch_entries}")
        set(TORCH_CUDA_ARCH_LIST
            "${cellerator_tokenization_torch_arch_list}" CACHE STRING
            "CUDA architectures passed to the selected Torch package")
        set(ENV{TORCH_CUDA_ARCH_LIST}
            "${cellerator_tokenization_torch_arch_list}")
    endif()
endif()
find_package(Torch REQUIRED)
if(cellerator_tokenization_saved_cuda_architectures)
    set(CMAKE_CUDA_ARCHITECTURES
        "${cellerator_tokenization_saved_cuda_architectures}" CACHE STRING
        "CUDA architectures selected for Cellerator" FORCE)
endif()
if(TARGET torch_library)
    # The exported imported target carries the selected Caffe2 dependency
    # graph. TORCH_LIBRARIES may also contain stale find_library results from
    # a different package prefix and duplicates already present transitively.
    set(cellerator_tokenization_torch_link_target torch torch_library)
elseif(TARGET torch)
    set(cellerator_tokenization_torch_link_target torch)
else()
    set(cellerator_tokenization_torch_link_target ${TORCH_LIBRARIES})
endif()

# Record the selected package's runtime library directory for local execution.
if(CELLERATOR_ENABLE_PYTHON AND cellerator_tokenization_python_torch_prefix)
    set(cellerator_tokenization_torch_lib_dir
        "${cellerator_tokenization_python_torch_prefix}/../../lib")
    get_filename_component(cellerator_tokenization_torch_lib_dir
        "${cellerator_tokenization_torch_lib_dir}" REALPATH)
elseif(CELLERATOR_LIBTORCH_ROOT)
    set(cellerator_tokenization_torch_lib_dir
        "${CELLERATOR_LIBTORCH_ROOT}/lib")
elseif(TORCH_INSTALL_PREFIX)
    set(cellerator_tokenization_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")
endif()
if(cellerator_tokenization_torch_lib_dir)
endif()

# This component can provide CUDA append kernels independently of Product2's
# host binding. AUTO keeps a usable ATen CPU target when no CUDA compiler is
# configured; ON requires the CUDA toolchain.
if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF"
   AND NOT CMAKE_CUDA_COMPILER_LOADED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    elseif(CELLERATOR_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR
            "CELLERATOR_ENABLE_CUDA=ON requires a CUDA compiler for tokenization")
    endif()
endif()

add_library(cellerator_aten_runtime INTERFACE)
add_library(Cellerator::aten_runtime ALIAS cellerator_aten_runtime)
target_compile_features(cellerator_aten_runtime INTERFACE cxx_std_17)
target_link_libraries(cellerator_aten_runtime INTERFACE
    ${cellerator_tokenization_torch_link_target})
if(cellerator_tokenization_torch_lib_dir)
    set_property(TARGET cellerator_aten_runtime PROPERTY BUILD_RPATH
        "${cellerator_tokenization_torch_lib_dir}")
endif()
if(TORCH_CXX_FLAGS)
    separate_arguments(CELLERATOR_TOKENIZATION_TORCH_CXX_FLAGS
        NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(cellerator_aten_runtime INTERFACE
        ${CELLERATOR_TOKENIZATION_TORCH_CXX_FLAGS})
endif()

add_library(cellerator_aten_ops STATIC
    src/compute/operators/aten_ops.cc)
add_library(Cellerator::aten_ops ALIAS cellerator_aten_ops)
target_include_directories(cellerator_aten_ops PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_aten_ops PUBLIC Cellerator::aten_runtime)
target_compile_features(cellerator_aten_ops PUBLIC cxx_std_17)
set_target_properties(cellerator_aten_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_tokenization STATIC
    src/compute/tokenization/tokenization.cc
    src/compute/tokenization/gene_annotation.cc
    src/compute/tokenization/gene_vocabulary.cc
    src/compute/tokenization/gene_tokenizer.cc
    src/compute/tokenization/sequence_buffer.cc)
add_library(Cellerator::tokenization ALIAS cellerator_tokenization)
target_include_directories(cellerator_tokenization PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_tokenization PUBLIC
    Cellerator::aten_ops Cellerator::aten_runtime)
target_compile_features(cellerator_tokenization PUBLIC cxx_std_17)
set_target_properties(cellerator_tokenization PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF" AND
   CMAKE_CUDA_COMPILER_LOADED AND
   EXISTS "${PROJECT_SOURCE_DIR}/src/compute/tokenization/append.cu")
    target_sources(cellerator_tokenization PRIVATE
        src/compute/tokenization/append.cu)
    target_compile_definitions(cellerator_tokenization PRIVATE
        CELLERATOR_TOKENIZATION_HAS_CUDA=1)
    target_link_libraries(cellerator_tokenization PUBLIC CUDA::cudart)
    set_target_properties(cellerator_tokenization PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    if(CELLERATOR_ENABLE_CUDA_LINEINFO)
        target_compile_options(cellerator_tokenization PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
endif()

set_property(TARGET cellerator_aten_runtime PROPERTY EXPORT_NAME aten_runtime)
set_property(TARGET cellerator_aten_ops PROPERTY EXPORT_NAME aten_ops)
set_property(TARGET cellerator_tokenization PROPERTY EXPORT_NAME tokenization)
install(TARGETS cellerator_aten_runtime cellerator_aten_ops
                cellerator_tokenization
    EXPORT CelleratorTokenizationTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization)
install(EXPORT CelleratorTokenizationTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    COMPONENT Tokenization)

if(EXISTS "${PROJECT_SOURCE_DIR}/include/Cellerator/compute/operators/aten_ops.hh")
    install(FILES include/Cellerator/compute/operators/aten_ops.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/compute/operators
        COMPONENT Tokenization)
endif()
install(DIRECTORY include/Cellerator/compute/tokenization
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/compute
    COMPONENT Tokenization)

# Extend the package config produced by the host SDK so downstream C++ users
# can request either the numerical operations or the full tokenizer facility.
if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
    set(cellerator_tokenization_cuda_dependency "")
    if(CMAKE_CUDA_COMPILER_LOADED)
        set(cellerator_tokenization_cuda_dependency
            "  find_dependency(CUDAToolkit REQUIRED)\n")
    endif()
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "if(\"aten_ops\" IN_LIST Cellerator_FIND_COMPONENTS OR \"tokenization\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "  include(CMakeFindDependencyMacro)\n"
        "  find_dependency(Torch REQUIRED)\n"
        "${cellerator_tokenization_cuda_dependency}"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTokenizationTargets.cmake\")\n"
        "  set(Cellerator_aten_ops_FOUND TRUE)\n"
        "  set(Cellerator_tokenization_FOUND TRUE)\n"
        "endif()\n"
        "foreach(_cellerator_component IN LISTS Cellerator_FIND_COMPONENTS)\n"
        "  if(Cellerator_FIND_REQUIRED_\${_cellerator_component} AND NOT Cellerator_\${_cellerator_component}_FOUND)\n"
        "    set(Cellerator_FOUND FALSE)\n"
        "    set(Cellerator_NOT_FOUND_MESSAGE \"Required Cellerator component '\${_cellerator_component}' is unavailable\")\n"
        "  endif()\n"
        "endforeach()\n")
endif()

if(CELLERATOR_ENABLE_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / 'lib')"
        RESULT_VARIABLE cellerator_tokenization_python_lib_status
        OUTPUT_VARIABLE cellerator_tokenization_python_lib_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(cellerator_tokenization_python_lib_status EQUAL 0)
        find_library(CELLERATOR_TOKENIZATION_TORCH_PYTHON_LIBRARY
            NAMES torch_python
            HINTS "${cellerator_tokenization_python_lib_dir}"
            NO_DEFAULT_PATH)
    endif()
    if(NOT CELLERATOR_TOKENIZATION_TORCH_PYTHON_LIBRARY)
        message(FATAL_ERROR
            "The optional cellerator.tokenization extension requires torch_python "
            "from the Torch installation selected by Python")
    endif()
    pybind11_add_module(cellerator_python_tokenization MODULE
        bindings/tokenization/module.cc
        bindings/tokenization/gene_tokenization.cc)
    set_target_properties(cellerator_python_tokenization PROPERTIES
        OUTPUT_NAME _tokenization
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        BUILD_RPATH "${cellerator_tokenization_python_lib_dir}"
        INSTALL_RPATH "$ORIGIN/../torch/lib")
    target_include_directories(cellerator_python_tokenization PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cellerator_python_tokenization PRIVATE
        Cellerator::tokenization
        ${CELLERATOR_TOKENIZATION_TORCH_PYTHON_LIBRARY})
    install(TARGETS cellerator_python_tokenization
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions
            cellerator_python_tokenization)
    endif()
endif()

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${PROJECT_SOURCE_DIR}/tests/tokenization/numerics_test.cc")
        add_executable(cellerator_tokenization_numerics_test
            tests/tokenization/numerics_test.cc)
        target_link_libraries(cellerator_tokenization_numerics_test PRIVATE
            Cellerator::aten_ops)
        add_test(NAME cellerator_tokenization_numerics
            COMMAND cellerator_tokenization_numerics_test)
    endif()
    foreach(test_name IN ITEMS core append)
        if(EXISTS "${PROJECT_SOURCE_DIR}/tests/tokenization/${test_name}_test.cc")
            add_executable(cellerator_tokenization_${test_name}_test
                "tests/tokenization/${test_name}_test.cc")
            target_link_libraries(cellerator_tokenization_${test_name}_test PRIVATE
                Cellerator::tokenization)
            add_test(NAME cellerator_tokenization_${test_name}
                COMMAND cellerator_tokenization_${test_name}_test)
        endif()
    endforeach()
    if(EXISTS "${PROJECT_SOURCE_DIR}/tests/tokenization/gene_vocabulary_test.cc")
        add_executable(cellerator_gene_vocabulary_test
            tests/tokenization/gene_vocabulary_test.cc)
        target_link_libraries(cellerator_gene_vocabulary_test PRIVATE
            Cellerator::tokenization)
        add_test(NAME cellerator_gene_vocabulary
            COMMAND cellerator_gene_vocabulary_test)
    endif()
endif()
