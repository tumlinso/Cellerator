# Minimal CPU/CUDA-compatible biological identity and operand ABI. This target
# intentionally has no runtime or CUDA-library dependency so Baseplane can
# consume the contract without joining Cellerator's numerical runtime.
add_library(cellerator_biological_abi INTERFACE)
add_library(Cellerator::biological_abi ALIAS cellerator_biological_abi)
target_include_directories(cellerator_biological_abi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_biological_abi INTERFACE cxx_std_17)

add_library(cellerator_identity_registry STATIC
    src/execution/identity_registry.cc
)
add_library(Cellerator::identity_registry ALIAS cellerator_identity_registry)
target_include_directories(cellerator_identity_registry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_identity_registry PUBLIC
    Cellerator::biological_abi
)
target_compile_features(cellerator_identity_registry PUBLIC cxx_std_17)

# Unified CSG1/CPE2/CPK1 acquisition and projection activation. The planner-v2
# phase model remains the sole cost authority; this target only reports the
# acquisition observations consumed by it.
add_library(cellerator_geometry_acquisition STATIC
    src/execution/geometry_acquisition.cc
    src/execution/geometry_acquisition_diagnostics.cc
    src/execution/geometry_acquisition_v2/assembly.cc
    src/execution/geometry_acquisition_v2/external_payload.cc
    src/execution/geometry_acquisition_v2/projections.cc
    src/execution/geometry_acquisition_v2/schema.cc
    src/execution/projection_activation_v2.cc
)
add_library(Cellerator::geometry_acquisition ALIAS
    cellerator_geometry_acquisition)
target_include_directories(cellerator_geometry_acquisition PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_geometry_acquisition PUBLIC
    Cellerator::operation_core
    Cellerator::planner
    Cellerator::semantic_geometry_v1
    CellPack::execution_image_v2
)
target_compile_features(cellerator_geometry_acquisition PUBLIC cxx_std_17)
