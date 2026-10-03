# Source-tree contract seam. Installed component exports belong to CE-IS1-BUILD.
get_filename_component(_ce_is1_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET Cellerator::host_operation_contracts)
  add_library(cellerator_host_operation_contracts INTERFACE)
  add_library(Cellerator::host_operation_contracts ALIAS cellerator_host_operation_contracts)
  target_include_directories(cellerator_host_operation_contracts INTERFACE "${_ce_is1_root}/include")
  target_compile_features(cellerator_host_operation_contracts INTERFACE cxx_std_20)
endif()
unset(_ce_is1_root)
