find_library(CELLERATOR_TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH)
if(NOT CELLERATOR_TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR
        "The optional Python Torch adapter requires torch_python from the selected Torch installation")
endif()
pybind11_add_module(cellerator_python_torch MODULE
    "${CMAKE_CURRENT_LIST_DIR}/python_module.cc")
set_target_properties(cellerator_python_torch PROPERTIES
    OUTPUT_NAME _torch
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    INSTALL_RPATH "$ORIGIN/../.libs"
    BUILD_WITH_INSTALL_RPATH ON)
target_include_directories(cellerator_python_torch PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cellerator_python_torch PRIVATE
    Cellerator::torch
    Cellerator::indexed_mechanism
    ${CELLERATOR_TORCH_PYTHON_LIBRARY})
if(CELLERATOR_ENABLE_SPARSE_SUPPORT)
    target_sources(cellerator_python_torch PRIVATE
        ${PROJECT_SOURCE_DIR}/python/bindings/sparse_support/module.cc
        ${PROJECT_SOURCE_DIR}/python/bindings/sparse_support/indexing_bindings.cc
        ${PROJECT_SOURCE_DIR}/python/bindings/sparse_support/geometry_bindings.cc
        ${PROJECT_SOURCE_DIR}/python/bindings/sparse_support/prepared_routing.cc)
    target_link_libraries(cellerator_python_torch PRIVATE
        Cellerator::sparse_support Cellerator::sparse_support_numeric)
    target_compile_definitions(cellerator_python_torch PRIVATE
        CELLERATOR_HAS_SPARSE_SUPPORT=1)
endif()
install(TARGETS cellerator_python_torch
    LIBRARY DESTINATION cellerator/torch COMPONENT Python
    RUNTIME DESTINATION cellerator/torch COMPONENT Python)
if(TARGET cellerator_python_extensions)
    add_dependencies(cellerator_python_extensions cellerator_python_torch)
endif()
