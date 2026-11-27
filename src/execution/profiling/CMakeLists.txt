add_library(cellerator_profiling_v1 STATIC
    mechanism_manifest.cc
    preparation/execution_export_v2.cc
    partition_export.cc
    resource_receipt.cc
    static_markers.cc
)
add_library(Cellerator::profiling_v1 ALIAS cellerator_profiling_v1)
target_include_directories(cellerator_profiling_v1 PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_profiling_v1 PUBLIC cxx_std_20)
target_link_libraries(cellerator_profiling_v1 PRIVATE
    Cellerator::execution_record_validation)
