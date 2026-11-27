# Reference reduction over canonical, record, and tile layouts.
add_library(cellpack_feature_weighted_row_reduction STATIC
    ${PROJECT_SOURCE_DIR}/src/ops/reduction/feature_weighted_row_reduction.cc
)
add_library(CellPack::feature_weighted_row_reduction ALIAS
    cellpack_feature_weighted_row_reduction)
target_include_directories(cellpack_feature_weighted_row_reduction PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_feature_weighted_row_reduction PUBLIC
    CellPack::warp_tiles
    CellPack::apply_plan
)
set_target_properties(cellpack_feature_weighted_row_reduction PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_feature_weighted_row_reduction)

# Asynchronous reduction over the existing warp-tile layout.
add_library(cellpack_feature_weighted_row_reduction_cuda STATIC
    ${PROJECT_SOURCE_DIR}/src/ops/reduction/feature_weighted_row_reduction_cuda.cu
)
add_library(CellPack::feature_weighted_row_reduction_cuda ALIAS
    cellpack_feature_weighted_row_reduction_cuda)
target_include_directories(cellpack_feature_weighted_row_reduction_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_feature_weighted_row_reduction_cuda PUBLIC
    cellpack_feature_weighted_row_reduction
    CUDA::cudart
)
target_compile_features(cellpack_feature_weighted_row_reduction_cuda PUBLIC cxx_std_20)
set_target_properties(cellpack_feature_weighted_row_reduction_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_feature_weighted_row_reduction_cuda)
endif()
