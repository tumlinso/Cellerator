include(GNUInstallDirs)
get_filename_component(_sequence_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_library(cellerator_sequence_ops STATIC
    dna2.cc motif.cc predicate_plan.cc source.cc
    "${_sequence_root}/src/layout/sequence/dna2.cc")
add_library(Cellerator::sequence_ops ALIAS cellerator_sequence_ops)
target_compile_features(cellerator_sequence_ops PUBLIC cxx_std_20)
target_include_directories(cellerator_sequence_ops PUBLIC
    $<BUILD_INTERFACE:${_sequence_root}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(cellerator_sequence_ops PROPERTIES EXPORT_NAME sequence_ops POSITION_INDEPENDENT_CODE ON)

if(CELLERATOR_HAS_CUDA)
    add_library(cellerator_sequence_ops_cuda STATIC dna2.cu "${_sequence_root}/src/layout/sequence/dna2.cu")
    add_library(Cellerator::sequence_ops_cuda ALIAS cellerator_sequence_ops_cuda)
    target_link_libraries(cellerator_sequence_ops_cuda PUBLIC Cellerator::sequence_ops CUDA::cudart)
    set_target_properties(cellerator_sequence_ops_cuda PROPERTIES
        EXPORT_NAME sequence_ops_cuda CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
endif()
