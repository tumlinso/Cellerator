include(GNUInstallDirs)

add_library(cellerator_value_readiness STATIC value_readiness.cu)
add_library(Cellerator::value_readiness ALIAS cellerator_value_readiness)
target_include_directories(cellerator_value_readiness PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cellerator_value_readiness PUBLIC cxx_std_20)
target_link_libraries(cellerator_value_readiness PUBLIC CUDA::cudart)
set_target_properties(cellerator_value_readiness PROPERTIES
    EXPORT_NAME value_readiness
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
