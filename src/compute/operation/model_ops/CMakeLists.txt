add_library(cellerator_model_ops STATIC model_ops.cu)
add_library(Cellerator::model_ops ALIAS cellerator_model_ops)
target_include_directories(cellerator_model_ops PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cellerator_model_ops PUBLIC CUDA::cudart)
target_compile_features(cellerator_model_ops PUBLIC cxx_std_17)
set_target_properties(cellerator_model_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
