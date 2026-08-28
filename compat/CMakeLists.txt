# Compatibility targets are defined here as historical surfaces move out of
# canonical implementation ownership. Compatibility may depend on current
# Cellerator contracts; current Cellerator targets must not depend on evidence.

# Historical format comparison retained as evidence. Its CellShard blocked-ELL
# conversion dependency is private to CellShard and therefore deliberately
# quarantined here instead of being exposed through current interop headers.
add_executable(celleratorLegacyPreprocessFormatCompareBench
    legacy_sparse/bench/preprocess_format_compare_bench.cu
)
target_include_directories(celleratorLegacyPreprocessFormatCompareBench PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CellShard_SOURCE_DIR}/src
)
target_link_libraries(celleratorLegacyPreprocessFormatCompareBench PRIVATE
    Cellerator::preprocess
    CUDA::cudart
)
set_target_properties(celleratorLegacyPreprocessFormatCompareBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(celleratorLegacyPreprocessFormatCompareBench)
