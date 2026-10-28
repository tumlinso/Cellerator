get_filename_component(_ce_ops_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${_ce_ops_root}/src/ops/process" "${CMAKE_CURRENT_BINARY_DIR}/ops-process")
unset(_ce_ops_root)
