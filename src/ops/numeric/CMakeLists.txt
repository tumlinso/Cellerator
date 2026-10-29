get_filename_component(_ce_ops_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_ops_numeric STATIC local_arithmetic.cc)
add_library(Cellerator::ops_numeric ALIAS cellerator_ops_numeric)
target_include_directories(cellerator_ops_numeric PUBLIC
    $<BUILD_INTERFACE:${_ce_ops_root}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_ops_numeric PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ops_numeric PRIVATE -fno-fast-math -ffp-contract=off)
endif()
set_target_properties(cellerator_ops_numeric PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME ops_numeric)
unset(_ce_ops_root)
