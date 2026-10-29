include(GNUInstallDirs)

option(CELLERATOR_ENABLE_MODELING "Build host sparse modeling primitives" ${CELLERATOR_ENABLE_SPARSE_SUPPORT})
if(NOT CELLERATOR_ENABLE_MODELING)
    return()
endif()
if(NOT TARGET Cellerator::support OR NOT TARGET Cellerator::indexing)
    message(FATAL_ERROR "CELLERATOR_ENABLE_MODELING requires CELLERATOR_ENABLE_SPARSE_SUPPORT")
endif()

add_library(cellerator_modeling STATIC
    ${PROJECT_SOURCE_DIR}/src/modeling/sparse_relation.cc)
add_library(Cellerator::modeling ALIAS cellerator_modeling)
set_target_properties(cellerator_modeling PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXPORT_NAME modeling)
target_compile_features(cellerator_modeling PUBLIC cxx_std_17)
target_include_directories(cellerator_modeling PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_modeling PUBLIC Cellerator::support Cellerator::indexing)

if(CELLERATOR_ENABLE_PYTHON)
    # Python/pybind11 are already discovered by CelleratorPython.cmake.
    pybind11_add_module(cellerator_modeling_python MODULE
        ${PROJECT_SOURCE_DIR}/bindings/modeling/module.cc)
    target_link_libraries(cellerator_modeling_python PRIVATE Cellerator::modeling Python::Module)
    set_target_properties(cellerator_modeling_python PROPERTIES OUTPUT_NAME _modeling_native)
    install(TARGETS cellerator_modeling_python
        LIBRARY DESTINATION cellerator COMPONENT Python
        RUNTIME DESTINATION cellerator COMPONENT Python)
    if(TARGET cellerator_python_extensions)
        add_dependencies(cellerator_python_extensions cellerator_modeling_python)
    endif()
endif()

if(CELLERATOR_BUILD_TESTS)
    enable_testing()
    add_executable(cellerator_coupling_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/coupling_test.cc)
    target_link_libraries(cellerator_coupling_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_coupling_test COMMAND cellerator_coupling_test)
    add_executable(cellerator_reactions_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/reactions_test.cc)
    target_link_libraries(cellerator_reactions_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_reactions_test COMMAND cellerator_reactions_test)
    add_executable(cellerator_inheritance_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/inheritance_test.cc)
    target_link_libraries(cellerator_inheritance_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_inheritance_test COMMAND cellerator_inheritance_test)
    add_executable(cellerator_modeling_incremental_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/incremental_test.cc)
    target_link_libraries(cellerator_modeling_incremental_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_modeling_incremental_test COMMAND cellerator_modeling_incremental_test)
    add_executable(cellerator_accessibility_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/accessibility_test.cc)
    target_link_libraries(cellerator_accessibility_test PRIVATE Cellerator::modeling)
    target_compile_features(cellerator_accessibility_test PRIVATE cxx_std_20)
    add_test(NAME cellerator_accessibility_test COMMAND cellerator_accessibility_test)
    add_executable(cellerator_modeling_native_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/sparse_modeling_test.cc)
    target_link_libraries(cellerator_modeling_native_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_modeling_native_test COMMAND cellerator_modeling_native_test)
    add_executable(cellerator_entity_layout_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/entity_layout_test.cc)
    target_link_libraries(cellerator_entity_layout_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_entity_layout_test COMMAND cellerator_entity_layout_test)
    add_executable(cellerator_model_contract_test
        ${PROJECT_SOURCE_DIR}/tests/framework/native/model_contract_test.cc)
    target_link_libraries(cellerator_model_contract_test PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_model_contract_test COMMAND cellerator_model_contract_test)
    add_executable(cellerator_cell_relation_example
        ${PROJECT_SOURCE_DIR}/tests/framework/native/cell_relation_example.cc)
    target_link_libraries(cellerator_cell_relation_example PRIVATE Cellerator::modeling)
    add_test(NAME cellerator_cell_relation_example COMMAND cellerator_cell_relation_example)
    if(TARGET cellerator_modeling_python)
        add_test(NAME cellerator_modeling_binding_test
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=$<TARGET_FILE_DIR:cellerator_modeling_python>"
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/framework/native/test_native_binding.py)
    endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/Cellerator/modeling
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Cellerator)
install(TARGETS cellerator_modeling EXPORT CelleratorModelingTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorModelingTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
if(CELLERATOR_ENABLE_CUDA STREQUAL "OFF" AND DEFINED CELLERATOR_HOST_CONFIG)
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorModelingTargets.cmake\")\n"
        "set(Cellerator_modeling_FOUND TRUE)\n")
endif()
