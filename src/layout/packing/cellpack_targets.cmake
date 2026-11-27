add_library(cellpack_cell_block_records_cuda STATIC
    ${PROJECT_SOURCE_DIR}/src/layout/packing/cell_block_records_cuda.cu
)
add_library(CellPack::cell_block_records_cuda ALIAS cellpack_cell_block_records_cuda)
target_include_directories(cellpack_cell_block_records_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_cell_block_records_cuda PUBLIC CellPack::cellpack CUDA::cudart)
target_compile_features(cellpack_cell_block_records_cuda PUBLIC cxx_std_20)
set_target_properties(cellpack_cell_block_records_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_cell_block_records_cuda)
endif()

# Reference warp-tile construction and its packed layout contract.
add_library(cellpack_warp_tiles STATIC
    ${PROJECT_SOURCE_DIR}/src/layout/packing/warp_tiles.cc
)
add_library(CellPack::warp_tiles ALIAS cellpack_warp_tiles)
target_include_directories(cellpack_warp_tiles PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_warp_tiles PUBLIC
    CellPack::cellpack
    cellpack_local_cell_ordering
)
target_compile_features(cellpack_warp_tiles PUBLIC cxx_std_20)
set_target_properties(cellpack_warp_tiles PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_warp_tiles)
endif()

# Asynchronous CUDA construction of the same packed tile layout.
add_library(cellpack_warp_tiles_cuda STATIC
    ${PROJECT_SOURCE_DIR}/src/layout/packing/warp_tiles_cuda.cu
)
add_library(CellPack::warp_tiles_cuda ALIAS cellpack_warp_tiles_cuda)
target_include_directories(cellpack_warp_tiles_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_warp_tiles_cuda PUBLIC
    CellPack::warp_tiles
    CUDA::cudart
)
target_compile_features(cellpack_warp_tiles_cuda PUBLIC cxx_std_20)
set_target_properties(cellpack_warp_tiles_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_warp_tiles_cuda)
endif()

add_library(cellpack_apply_plan STATIC
    ${PROJECT_SOURCE_DIR}/src/layout/packing/apply_plan.cc
    ${PROJECT_SOURCE_DIR}/src/layout/packing/apply_plan_cuda.cu
)
add_library(CellPack::apply_plan ALIAS cellpack_apply_plan)
target_include_directories(cellpack_apply_plan PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_apply_plan PUBLIC CellPack::cellpack CUDA::cudart)
set_target_properties(cellpack_apply_plan PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_apply_plan)

# Validation and pointer rebinding for the pointer-free CPK1 image.
add_library(cellpack_persistent_packing_payload STATIC
    ${PROJECT_SOURCE_DIR}/src/layout/packing/persistence/persistent_packing_payload.cc
)
add_library(CellPack::persistent_packing_payload ALIAS
    cellpack_persistent_packing_payload)
target_include_directories(cellpack_persistent_packing_payload PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_persistent_packing_payload PUBLIC
    CellPack::feature_weighted_row_reduction
)
set_target_properties(cellpack_persistent_packing_payload PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_persistent_packing_payload)
