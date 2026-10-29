include(GNUInstallDirs)

option(CELLERATOR_ENABLE_CONDITIONAL "Build generic host expert-major sparse dispatch" ON)
if(NOT CELLERATOR_ENABLE_CONDITIONAL)
    return()
endif()

add_library(cellerator_conditional STATIC
    ${PROJECT_SOURCE_DIR}/src/execution/conditional/dispatch.cc
    ${PROJECT_SOURCE_DIR}/src/execution/conditional/parameter_store.cc)
add_library(Cellerator::conditional ALIAS cellerator_conditional)
set_target_properties(cellerator_conditional PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXPORT_NAME conditional)
target_compile_features(cellerator_conditional PUBLIC cxx_std_17)
target_include_directories(cellerator_conditional PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CELLERATOR_ENABLE_PYTHON)
    pybind11_add_module(cellerator_conditional_python MODULE
        ${PROJECT_SOURCE_DIR}/bindings/conditional/module.cc)
    target_link_libraries(cellerator_conditional_python PRIVATE Cellerator::conditional Python::Module)
    set_target_properties(cellerator_conditional_python PROPERTIES OUTPUT_NAME _conditional_native)
    install(TARGETS cellerator_conditional_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_conditional_python)
    endif()
endif()

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_executable(cellerator_conditional_scheduler_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/scheduler_test.cc)
    target_link_libraries(cellerator_conditional_scheduler_test PRIVATE Cellerator::conditional)
    add_test(NAME cellerator_conditional_scheduler_test COMMAND cellerator_conditional_scheduler_test)
    add_executable(cellerator_conditional_scheduler_example
        ${PROJECT_SOURCE_DIR}/examples/framework/native/scheduler_example.cc)
    target_link_libraries(cellerator_conditional_scheduler_example PRIVATE Cellerator::conditional)
    add_test(NAME cellerator_conditional_scheduler_example COMMAND cellerator_conditional_scheduler_example)
    find_package(Threads REQUIRED)
    add_executable(cellerator_parameter_store_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/parameter_store_test.cc)
    target_link_libraries(cellerator_parameter_store_test PRIVATE Cellerator::conditional Threads::Threads)
    add_test(NAME cellerator_parameter_store_test COMMAND cellerator_parameter_store_test)
    add_executable(cellerator_conditional_parameter_execution_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/conditional_parameter_execution_test.cc)
    target_link_libraries(cellerator_conditional_parameter_execution_test PRIVATE Cellerator::conditional Threads::Threads)
    add_test(NAME cellerator_conditional_parameter_execution_test COMMAND cellerator_conditional_parameter_execution_test)
    add_executable(cellerator_conditional_dispatch_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/conditional_dispatch_test.cc)
    target_link_libraries(cellerator_conditional_dispatch_test PRIVATE Cellerator::conditional)
    add_test(NAME cellerator_conditional_dispatch_test COMMAND cellerator_conditional_dispatch_test)
    add_executable(cellerator_conditional_dispatch_example
        ${PROJECT_SOURCE_DIR}/tests/framework/native/conditional_dispatch_example.cc)
    target_link_libraries(cellerator_conditional_dispatch_example PRIVATE Cellerator::conditional)
    add_test(NAME cellerator_conditional_dispatch_example COMMAND cellerator_conditional_dispatch_example)
    if(TARGET cellerator_conditional_python)
        add_test(NAME cellerator_conditional_engine_bridge_test
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$<TARGET_FILE_DIR:cellerator_conditional_python>"
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/framework/native/conditional_engine_bridge_test.py)
        add_test(NAME cellerator_conditional_binding_test
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:cellerator_conditional_python>"
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/framework/native/conditional_binding_test.py)
    endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/Cellerator/execution/conditional
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/execution)
install(TARGETS cellerator_conditional EXPORT CelleratorConditionalTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorConditionalTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF" AND DEFINED CELLERATOR_HOST_CONFIG)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorConditionalTargets.cmake\")\n"
        "set(Cellerator_conditional_FOUND TRUE)\n")
endif()
