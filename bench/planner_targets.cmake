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
    Cellerator::spmm_reference
)
target_link_libraries(celleratorCeArch92RealRegimeBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    Cellerator::spmm_reference
)
set_target_properties(celleratorCeArch76CandidateBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
set_target_properties(celleratorCeArch92RealRegimeBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

# CP-MATH-03 benchmark promoted from its standalone validation command so
# background evidence can rebuild it from an immutable source snapshot.
add_executable(cpMathPackedDenseOperandBench
    bench/math/packed_dense_operand_bench.cu
)
target_include_directories(cpMathPackedDenseOperandBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)
target_link_libraries(cpMathPackedDenseOperandBench PRIVATE
    Cellerator::packed_dense_operand
)
set_target_properties(cpMathPackedDenseOperandBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cpMathPackedDenseOperandBench)
