set(_cellerator_native_data_test_targets)

function(cellerator_add_native_data_test target test_name source link_target standard)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/${source}")
    target_compile_features(${target} PRIVATE cxx_std_${standard})
    target_link_libraries(${target} PRIVATE "${link_target}")
    add_test(NAME "native_data_${test_name}" COMMAND ${target})
    set_tests_properties("native_data_${test_name}" PROPERTIES
        LABELS "native_data" TIMEOUT 60)
    set(_cellerator_native_data_test_targets
        ${_cellerator_native_data_test_targets} ${target} PARENT_SCOPE)
endfunction()

cellerator_add_native_data_test(
    cellerator_native_data_gated_product_test gated_product
    tests/native_data/gated_product_host_test.cc
    Cellerator::gated_product 20)
cellerator_add_native_data_test(
    cellerator_native_data_gated_product_binding_test gated_product_binding
    tests/native_data/gated_product_binding_test.cc
    Cellerator::gated_product 20)
cellerator_add_native_data_test(
    cellerator_native_data_geometry_test geometry
    tests/native_data/dependency_geometry_test.cc
    Cellerator::dependency_geometry 20)
cellerator_add_native_data_test(
    cellerator_native_data_cartesian_access_test cartesian_access
    tests/native_data/cartesian_access_test.cc
    Cellerator::dependency_geometry 20)
cellerator_add_native_data_test(
    cellerator_native_data_foundation_test foundation
    tests/native_data/dependency_foundation_test.cc
    Cellerator::dependency_geometry 20)

add_executable(cellerator_native_data_reference_test
    "${PROJECT_SOURCE_DIR}/tests/native_data/reference/native_data_reference_test.cc")
target_include_directories(cellerator_native_data_reference_test PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/native_data/reference")
target_compile_features(cellerator_native_data_reference_test PRIVATE cxx_std_20)
target_link_libraries(cellerator_native_data_reference_test PRIVATE
    Cellerator::gated_product Cellerator::dependency_geometry)
add_test(NAME native_data_reference COMMAND cellerator_native_data_reference_test)
set_tests_properties(native_data_reference PROPERTIES
    LABELS "native_data" TIMEOUT 60)
list(APPEND _cellerator_native_data_test_targets
    cellerator_native_data_reference_test)

cellerator_add_native_data_test(
    cellerator_native_data_block_masked_test block_masked
    tests/tensor/block_masked_test.cc Cellerator::block_masked 17)
cellerator_add_native_data_test(
    cellerator_native_data_dependency_execution_test dependency_execution
    tests/execution_dependency/distributed_binding_test.cc
    Cellerator::dependency_execution 20)

add_executable(cellerator_native_data_condition_index_test
    "${PROJECT_SOURCE_DIR}/tests/native_data/condition_index_test.cc")
target_compile_features(cellerator_native_data_condition_index_test PRIVATE cxx_std_20)
target_link_libraries(cellerator_native_data_condition_index_test PRIVATE
    Cellerator::dependency_execution Cellerator::gated_product)
add_test(NAME native_data_condition_index
    COMMAND cellerator_native_data_condition_index_test)
set_tests_properties(native_data_condition_index PROPERTIES
    LABELS "native_data" TIMEOUT 60)
list(APPEND _cellerator_native_data_test_targets
    cellerator_native_data_condition_index_test)

if(TARGET Cellerator::modeling)
    function(cellerator_add_native_data_modeling_test target test_name source)
        add_executable(${target} "${PROJECT_SOURCE_DIR}/${source}")
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_link_libraries(${target} PRIVATE
            Cellerator::native_data Cellerator::modeling)
        add_test(NAME "native_data_${test_name}" COMMAND ${target})
        set_tests_properties("native_data_${test_name}" PROPERTIES
            LABELS "native_data" TIMEOUT 60)
        set(_cellerator_native_data_test_targets
            ${_cellerator_native_data_test_targets} ${target} PARENT_SCOPE)
    endfunction()

    cellerator_add_native_data_modeling_test(
        cellerator_native_data_sparse_modeling_test sparse_modeling
        tests/framework/native/sparse_modeling_test.cc)
    cellerator_add_native_data_modeling_test(
        cellerator_native_data_incremental_test incremental
        tests/framework/native/incremental_test.cc)
    cellerator_add_native_data_modeling_test(
        cellerator_native_data_cell_relation_example cell_relation_example
        tests/framework/native/cell_relation_example.cc)
endif()

add_custom_target(cellerator_native_data_tests
    DEPENDS ${_cellerator_native_data_test_targets})
