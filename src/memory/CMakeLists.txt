include(GNUInstallDirs)

add_library(cellerator_memory_contract STATIC
    allocation.cc
    generation_marks.cc
    image.cc
    cuda/copy.cu
)
add_library(Cellerator::memory_contract ALIAS cellerator_memory_contract)
target_include_directories(cellerator_memory_contract PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cellerator_memory_contract PUBLIC cxx_std_20)
target_link_libraries(cellerator_memory_contract PUBLIC CUDA::cudart)
set_target_properties(cellerator_memory_contract PROPERTIES
    EXPORT_NAME memory_contract
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
