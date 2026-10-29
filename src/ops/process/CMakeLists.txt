get_filename_component(_ce_ops_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_op_process_ports STATIC ports.cc ports_provider.cc)
add_library(Cellerator::op_process_ports ALIAS cellerator_op_process_ports)
add_library(Cellerator::private_ports ALIAS cellerator_op_process_ports)
target_include_directories(cellerator_op_process_ports PUBLIC
    $<BUILD_INTERFACE:${_ce_ops_root}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(cellerator_op_process_ports PROPERTIES EXPORT_NAME private_ports)
target_compile_features(cellerator_op_process_ports PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_op_process_ports PRIVATE -fno-fast-math -ffp-contract=off)
endif()
unset(_ce_ops_root)
