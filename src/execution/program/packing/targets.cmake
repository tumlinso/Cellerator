add_library(cellpack_gating_cuda STATIC
    ${PROJECT_SOURCE_DIR}/src/execution/program/packing/gating_cuda.cu
)
add_library(CellPack::gating_cuda ALIAS cellpack_gating_cuda)
target_include_directories(cellpack_gating_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellpack_gating_cuda PUBLIC CellPack::cellpack CUDA::cudart)
target_compile_features(cellpack_gating_cuda PUBLIC cxx_std_20)
set_target_properties(cellpack_gating_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
if(COMMAND cellerator_enable_perf)
    cellerator_enable_perf(cellpack_gating_cuda)
endif()
