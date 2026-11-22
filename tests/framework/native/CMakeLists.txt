function(cellerator_add_native_host_check target source owner)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${owner})
    if(CELLERATOR_BUILD_TESTS)
        add_test(NAME ${target} COMMAND ${target})
    endif()
endfunction()

cellerator_add_native_host_check(cellerator_coupling_test coupling_test.cc Cellerator::exchange)
cellerator_add_native_host_check(cellerator_reactions_test reactions_test.cc Cellerator::reaction)
cellerator_add_native_host_check(cellerator_inheritance_test inheritance_test.cc Cellerator::inheritance)
cellerator_add_native_host_check(cellerator_modeling_incremental_test incremental_test.cc Cellerator::dependency_execution)
cellerator_add_native_host_check(cellerator_accessibility_test accessibility_test.cc Cellerator::accessibility)
cellerator_add_native_host_check(cellerator_modeling_native_test sparse_modeling_test.cc Cellerator::dependency_execution)
cellerator_add_native_host_check(cellerator_entity_layout_test entity_layout_test.cc Cellerator::layout)
cellerator_add_native_host_check(cellerator_cell_relation_example cell_relation_example.cc Cellerator::dependency_execution)

if(CELLERATOR_BUILD_TESTS AND TARGET cellerator_modeling_python)
    add_test(NAME cellerator_modeling_binding_test
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:cellerator_modeling_python>"
            ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_native_binding.py)
endif()
