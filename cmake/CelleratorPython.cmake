include(GNUInstallDirs)

if(NOT TARGET Cellerator::product2)
    message(FATAL_ERROR "CELLERATOR_ENABLE_PYTHON requires CELLERATOR_BUILD_PRODUCT2")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cellerator_python_native MODULE
    ${PROJECT_SOURCE_DIR}/bindings/python/module.cc)
set_target_properties(cellerator_python_native PROPERTIES
    OUTPUT_NAME _native
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    BUILD_RPATH "$ORIGIN/.libs;$<TARGET_FILE_DIR:cellerator_product2>"
    INSTALL_RPATH "$ORIGIN/.libs")
target_link_libraries(cellerator_python_native PRIVATE
    Cellerator::product2
    Python::Module)
target_include_directories(cellerator_python_native PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
install(TARGETS cellerator_python_native
    LIBRARY DESTINATION cellerator COMPONENT Python
    RUNTIME DESTINATION cellerator COMPONENT Python)
add_custom_target(cellerator_python_extensions
    DEPENDS cellerator_python_native)

# Called only after the CUDA indexed-mechanism owner exists. Product2 remains
# available in the CUDA-off host build, with an explicit false capability.
function(cellerator_configure_python_mechanism_bindings)
    if(TARGET Cellerator::indexed_mechanism)
        target_compile_definitions(cellerator_python_native PRIVATE
            CELLERATOR_HAS_INDEXED_MECHANISM=1)
        target_link_libraries(cellerator_python_native PRIVATE
            Cellerator::indexed_mechanism)
        if(CELLERATOR_BUILD_TESTS AND
           EXISTS "${PROJECT_SOURCE_DIR}/tests/bindings/native/mechanism_handle_test.cc")
            add_executable(cellerator_indexed_mechanism_binding_test
                ${PROJECT_SOURCE_DIR}/tests/bindings/native/mechanism_handle_test.cc)
            target_link_libraries(cellerator_indexed_mechanism_binding_test PRIVATE
                Cellerator::indexed_mechanism)
            target_compile_features(cellerator_indexed_mechanism_binding_test PRIVATE
                cxx_std_20)
        endif()
    endif()
endfunction()
