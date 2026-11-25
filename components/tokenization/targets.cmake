include(GNUInstallDirs)

if(NOT CELLERATOR_ENABLE_TOKENIZATION)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/CelleratorTorchDependency.cmake")

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
target_compile_features(cellerator_aten_runtime INTERFACE cxx_std_20)
target_link_libraries(cellerator_aten_runtime INTERFACE
    ${CELLERATOR_TORCH_LINK_TARGETS})
if(CELLERATOR_TORCH_LIB_DIR)
    set_property(TARGET cellerator_aten_runtime PROPERTY BUILD_RPATH
        "${CELLERATOR_TORCH_LIB_DIR}")
endif()
if(CELLERATOR_TORCH_CXX_OPTIONS)
    target_compile_options(cellerator_aten_runtime INTERFACE
        ${CELLERATOR_TORCH_CXX_OPTIONS})
endif()

add_library(cellerator_aten_ops STATIC
    "${CMAKE_CURRENT_LIST_DIR}/src/aten_ops.cc")
add_library(Cellerator::aten_ops ALIAS cellerator_aten_ops)
target_include_directories(cellerator_aten_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_aten_ops PUBLIC Cellerator::headers Cellerator::aten_runtime)
target_compile_features(cellerator_aten_ops PUBLIC cxx_std_20)
set_target_properties(cellerator_aten_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(cellerator_tokenization STATIC
    "${CMAKE_CURRENT_LIST_DIR}/src/tokenization.cc"
    "${CMAKE_CURRENT_LIST_DIR}/src/gene_vocabulary_input.cc"
    "${CMAKE_CURRENT_LIST_DIR}/src/gene_vocabulary.cc"
    "${CMAKE_CURRENT_LIST_DIR}/src/gene_tokenizer.cc"
    "${CMAKE_CURRENT_LIST_DIR}/src/sequence_buffer.cc")
add_library(Cellerator::tokenization ALIAS cellerator_tokenization)
target_include_directories(cellerator_tokenization PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_tokenization PUBLIC
    Cellerator::headers
    Cellerator::aten_ops Cellerator::aten_runtime Cellerator::annotation)
target_compile_features(cellerator_tokenization PUBLIC cxx_std_20)
set_target_properties(cellerator_tokenization PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT CELLERATOR_ENABLE_CUDA STREQUAL "OFF" AND
   CMAKE_CUDA_COMPILER_LOADED AND
   EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/append.cu")
    target_sources(cellerator_tokenization PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/append.cu")
    target_compile_definitions(cellerator_tokenization PRIVATE
        CELLERATOR_TOKENIZATION_HAS_CUDA=1)
    target_link_libraries(cellerator_tokenization PUBLIC CUDA::cudart)
    set_target_properties(cellerator_tokenization PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    if(CELLERATOR_ENABLE_CUDA_LINEINFO)
        target_compile_options(cellerator_tokenization PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
endif()

set_property(TARGET cellerator_aten_runtime PROPERTY EXPORT_NAME aten_runtime)
set_property(TARGET cellerator_aten_ops PROPERTY EXPORT_NAME aten_ops)
set_property(TARGET cellerator_tokenization PROPERTY EXPORT_NAME tokenization)
if(CELLERATOR_ENABLE_MODULES)
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/../../cmake/CelleratorModuleInterfaces.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator COMPONENT Tokenization)
    add_library(cellerator_tokenizers_interface STATIC)
    set_target_properties(cellerator_tokenizers_interface PROPERTIES EXPORT_NAME tokenizers_interface_binary CXX_EXTENSIONS OFF)
    target_compile_features(cellerator_tokenizers_interface PUBLIC cxx_std_20)
    target_include_directories(cellerator_tokenizers_interface PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(cellerator_tokenizers_interface PUBLIC Cellerator::headers Cellerator::aten_runtime Cellerator::annotation)
    target_sources(cellerator_tokenizers_interface PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src"
        FILES "${CMAKE_CURRENT_LIST_DIR}/src/cellerator.tokenizers.ccm")
    target_link_libraries(cellerator_tokenization PUBLIC $<BUILD_INTERFACE:cellerator_tokenizers_interface>)
    install(TARGETS cellerator_tokenizers_interface EXPORT CelleratorTokenizersModuleTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization
        FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules COMPONENT Tokenization)
    install(EXPORT CelleratorTokenizersModuleTargets NAMESPACE Cellerator::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
        CXX_MODULES_DIRECTORY tokenizer-modules COMPONENT Tokenization)
endif()

install(TARGETS cellerator_aten_runtime cellerator_aten_ops
    EXPORT CelleratorAtenOpsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization)
install(EXPORT CelleratorAtenOpsTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator COMPONENT Tokenization)
install(TARGETS cellerator_tokenization
    EXPORT CelleratorTokenizationTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Tokenization)
install(EXPORT CelleratorTokenizationTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    COMPONENT Tokenization)

install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/Cellerator/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" COMPONENT Tokenization)

# Extend the package config produced by the host SDK so downstream C++ users
# can request either the numerical operations or the full tokenizer facility.
function(cellerator_write_tokenization_package_config)
    set(cellerator_tokenization_cuda_dependency "")
    if(CMAKE_CUDA_COMPILER_LOADED)
        set(cellerator_tokenization_cuda_dependency
            "  find_dependency(CUDAToolkit REQUIRED)\n")
    endif()
    set(tokenization_config "${CMAKE_CURRENT_BINARY_DIR}/TokenizationComponent.cmake")
    file(WRITE "${tokenization_config}"
        "include(CMakeFindDependencyMacro)\n"
        "find_dependency(Torch REQUIRED)\n"
        "${cellerator_tokenization_cuda_dependency}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorAtenOpsTargets.cmake\")\n"
        "set(Cellerator_aten_ops_FOUND TRUE)\n"
        "if(\"tokenization\" IN_LIST Cellerator_FIND_COMPONENTS AND TARGET Cellerator::annotation)\n"
        "  include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTokenizationTargets.cmake\")\n"
        "  if(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTokenizersModuleTargets.cmake\")\n"
        "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTokenizersModuleTargets.cmake\")\n"
        "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake\")\n"
        "    cellerator_attach_module(Cellerator::tokenization Cellerator::tokenizers_interface_binary)\n"
        "  endif()\n"
        "  set(Cellerator_tokenization_FOUND TRUE)\n"
        "endif()\n")
    install(FILES "${tokenization_config}" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator COMPONENT Tokenization)
endfunction()

cellerator_write_tokenization_package_config()

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
        ${PROJECT_SOURCE_DIR}/python/bindings/tokenization/module.cc
        ${PROJECT_SOURCE_DIR}/python/bindings/tokenization/gene_tokenization.cc)
    set_target_properties(cellerator_python_tokenization PROPERTIES
        OUTPUT_NAME _tokenization
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
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
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/numerics_test.cc")
        add_executable(cellerator_tokenization_numerics_test
            "${CMAKE_CURRENT_LIST_DIR}/tests/numerics_test.cc")
        target_link_libraries(cellerator_tokenization_numerics_test PRIVATE
            Cellerator::aten_ops)
        add_test(NAME cellerator_tokenization_numerics
            COMMAND cellerator_tokenization_numerics_test)
    endif()
    foreach(test_name IN ITEMS core append)
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/${test_name}_test.cc")
            add_executable(cellerator_tokenization_${test_name}_test
                "${CMAKE_CURRENT_LIST_DIR}/tests/${test_name}_test.cc")
            target_link_libraries(cellerator_tokenization_${test_name}_test PRIVATE
                Cellerator::tokenization)
            add_test(NAME cellerator_tokenization_${test_name}
                COMMAND cellerator_tokenization_${test_name}_test)
        endif()
    endforeach()
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/gene_vocabulary_test.cc")
        add_executable(cellerator_gene_vocabulary_test
            "${CMAKE_CURRENT_LIST_DIR}/tests/gene_vocabulary_test.cc")
        target_link_libraries(cellerator_gene_vocabulary_test PRIVATE
            Cellerator::tokenization)
        add_test(NAME cellerator_gene_vocabulary
            COMMAND cellerator_gene_vocabulary_test)
    endif()
endif()
