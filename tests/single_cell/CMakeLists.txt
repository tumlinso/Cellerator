set(CMAKE_CXX_SCAN_FOR_MODULES ON)
add_executable(cellerator_single_cell_methods_test methods.cpp)
target_link_libraries(cellerator_single_cell_methods_test PRIVATE
    Cellerator::preprocess Cellerator::velocity)
add_test(NAME cellerator-single-cell-methods COMMAND cellerator_single_cell_methods_test)

if(TARGET Cellerator::anndata AND (ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR))
    add_executable(cellerator_single_cell_anndata_test anndata_modeling.cpp)
    target_include_directories(cellerator_single_cell_anndata_test PRIVATE
        "${PROJECT_SOURCE_DIR}/components/preprocess/examples")
    target_link_libraries(cellerator_single_cell_anndata_test PRIVATE
        Cellerator::anndata Cellerator::preprocess Cellerator::velocity)
    add_test(NAME cellerator-single-cell-anndata COMMAND cellerator_single_cell_anndata_test)
endif()
