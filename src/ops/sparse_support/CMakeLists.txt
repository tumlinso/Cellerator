add_library(cellerator_ops_sparse_support STATIC
    training.cc training.cu prepared_routing.cc prepared_routing.cu)
add_library(Cellerator::ops_sparse_support ALIAS cellerator_ops_sparse_support)
add_library(Cellerator::sparse_support_numeric ALIAS cellerator_ops_sparse_support)
target_include_directories(cellerator_ops_sparse_support PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_ops_sparse_support PUBLIC
    Cellerator::indexed_mechanism Cellerator::architecture_provider
    Cellerator::sparse_support_cuda CUDA::cudart)
if(TARGET Cellerator::ce_geo_catalog)
    target_link_libraries(cellerator_ops_sparse_support PUBLIC
        Cellerator::ce_geo_catalog)
endif()
target_compile_features(cellerator_ops_sparse_support PUBLIC cxx_std_20)
set_target_properties(cellerator_ops_sparse_support PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    EXPORT_NAME sparse_support_numeric)
