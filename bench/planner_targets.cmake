add_executable(celleratorCeArch76CandidateBench
    bench/math/feature_major_candidate_compare_bench.cu
)
add_executable(celleratorCeArch92RealRegimeBench
    bench/architecture_evidence/real_regime_bench.cu
)
target_link_libraries(celleratorCeArch76CandidateBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    cellerator_math_v1_evidence
)
target_link_libraries(celleratorCeArch92RealRegimeBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    cellerator_math_v1_evidence
)
set_target_properties(celleratorCeArch76CandidateBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
set_target_properties(celleratorCeArch92RealRegimeBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
