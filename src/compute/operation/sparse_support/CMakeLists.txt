add_library(cellerator_sparse_support_numeric STATIC training.cc training.cu)
add_library(Cellerator::sparse_support_numeric ALIAS
    cellerator_sparse_support_numeric)
target_include_directories(cellerator_sparse_support_numeric PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_sparse_support_numeric PUBLIC
    Cellerator::indexed_mechanism Cellerator::architecture_provider CUDA::cudart)
if(TARGET Cellerator::ce_geo_catalog)
    target_link_libraries(cellerator_sparse_support_numeric PUBLIC
        Cellerator::ce_geo_catalog)
endif()
target_compile_features(cellerator_sparse_support_numeric PUBLIC cxx_std_20)
set_target_properties(cellerator_sparse_support_numeric PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
