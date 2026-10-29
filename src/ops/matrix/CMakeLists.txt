get_filename_component(_ce_ops_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_op_matrix_patch STATIC patch.cc patch_provider.cc)
add_library(Cellerator::op_matrix_patch ALIAS cellerator_op_matrix_patch)
add_library(Cellerator::matrix_patch ALIAS cellerator_op_matrix_patch)
target_include_directories(cellerator_op_matrix_patch PUBLIC
    $<BUILD_INTERFACE:${_ce_ops_root}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(cellerator_op_matrix_patch PROPERTIES EXPORT_NAME matrix_patch)
target_compile_features(cellerator_op_matrix_patch PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_op_matrix_patch PRIVATE -fno-fast-math -ffp-contract=off)
endif()
unset(_ce_ops_root)
