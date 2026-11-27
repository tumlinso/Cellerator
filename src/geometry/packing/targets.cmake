# Bounded local cell ordering for a frozen packing plan.
add_library(cellpack_local_cell_ordering STATIC
    ${PROJECT_SOURCE_DIR}/src/geometry/packing/local_cell_ordering.cc
    ${PROJECT_SOURCE_DIR}/src/geometry/packing/local_cell_ordering_cuda.cu
)
add_library(CellPack::local_cell_ordering ALIAS cellpack_local_cell_ordering)
target_include_directories(cellpack_local_cell_ordering PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(cellpack_local_cell_ordering PUBLIC CellPack::cellpack CUDA::cudart)
set_target_properties(cellpack_local_cell_ordering PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellpack_local_cell_ordering)
