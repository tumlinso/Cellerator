# Provider-specific layout construction consumes semantic geometry and the
# reusable physical projection owner without moving discovery into layout.
add_library(cellerator_semantic_projection_layout STATIC
    src/layout/mma_physical_work_layout.cc
    src/layout/mma_residual_builder.cc)
add_library(Cellerator::semantic_projection_layout ALIAS
    cellerator_semantic_projection_layout)
target_include_directories(cellerator_semantic_projection_layout PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_semantic_projection_layout PUBLIC
    Cellerator::layout Cellerator::semantic_geometry_v1
    CellPack::execution_image_v2)
target_compile_features(cellerator_semantic_projection_layout PUBLIC cxx_std_20)

# CE-ARCH-75: a second projection of the frozen CP-BP semantic structure.
# Construction and value packing are explicit host preparation; prepared
# execution consumes the native feature-major payload directly for N=1..16.
add_library(cellerator_feature_major_projection STATIC
    src/layout/projection/feature_major.cc
)
add_library(Cellerator::feature_major_projection ALIAS
    cellerator_feature_major_projection)
target_include_directories(cellerator_feature_major_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_feature_major_projection PUBLIC
    CellPack::persistent_packing_payload
    Cellerator::biological_abi
)
target_compile_features(cellerator_feature_major_projection PUBLIC cxx_std_20)

add_library(cellerator_transpose_projection STATIC
    src/layout/projection/transpose.cc
)
add_library(Cellerator::transpose_projection ALIAS
    cellerator_transpose_projection)
target_include_directories(cellerator_transpose_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_transpose_projection PUBLIC
    Cellerator::feature_major_projection
    Cellerator::biological_abi
)
target_compile_features(cellerator_transpose_projection PUBLIC cxx_std_20)
