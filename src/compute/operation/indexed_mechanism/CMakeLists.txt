add_library(cellerator_indexed_mechanism STATIC incidence.cc evaluators.cc evaluators.cu training.cu)
add_library(Cellerator::indexed_mechanism ALIAS cellerator_indexed_mechanism)
target_include_directories(cellerator_indexed_mechanism PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_indexed_mechanism PUBLIC CUDA::cudart CUDA::cuda_driver Cellerator::runtime Cellerator::executable_program)
set_target_properties(cellerator_indexed_mechanism PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_indexed_mechanism PUBLIC cxx_std_20)
set_target_properties(cellerator_indexed_mechanism PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
