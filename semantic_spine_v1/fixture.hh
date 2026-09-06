#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace spine_demo {
// An illustrative signed regulator-to-gene relation, not a fitted biological model.
inline constexpr std::size_t regulators = 4, genes = 5, edges = 9;
inline constexpr const char* regulator_names[] = {"Regulator-A", "Regulator-B", "Regulator-C", "Regulator-D"};
inline constexpr const char* gene_names[] = {"Gene-A", "Gene-B", "Gene-C", "Gene-D", "Gene-E"};
// Gene-D has no incoming edges. The non-square shape exposes direction errors.
inline constexpr std::uint32_t row_offsets[] = {0, 2, 4, 6, 6, 9};
inline constexpr std::uint32_t sources[] = {0, 2, 1, 3, 0, 3, 0, 1, 2};
inline constexpr float weights_1[] = {0.5f,-0.25f,1.0f,0.125f,-0.5f,0.75f,0.25f,-1.0f,0.5f};
inline constexpr float weights_2[] = {1.0f,-0.25f,0.5f,0.125f,-0.5f,0.25f,0.25f,-1.0f,0.5f};
// Exact binary16 encodings of the dyadic values above. No precision mismatch in the oracle.
inline constexpr std::uint16_t half_weights_1[] = {0x3800,0xb400,0x3c00,0x3000,0xb800,0x3a00,0x3400,0xbc00,0x3800};
inline constexpr std::uint16_t half_weights_2[] = {0x3c00,0xb400,0x3800,0x3000,0xb800,0x3400,0x3400,0xbc00,0x3800};
inline constexpr float state_a[] = {2.0f,1.0f,4.0f,2.0f};
inline constexpr float state_b[] = {1.0f,3.0f,2.0f,4.0f};
inline constexpr float gene_signal[] = {1.0f,-2.0f,0.5f,3.0f,2.0f};

inline void forward_reference(const float* weights,const float* input,double* output) {
    for (std::size_t d=0; d<genes; ++d) {
        double sum=0;
        for (auto e=row_offsets[d]; e<row_offsets[d+1]; ++e)
            sum+=static_cast<double>(weights[e])*input[sources[e]];
        output[d]=sum;
    }
}
inline void transpose_reference(const float* weights,const float* input,double* output) {
    for (std::size_t s=0; s<regulators; ++s) output[s]=0;
    for (std::size_t d=0; d<genes; ++d)
        for (auto e=row_offsets[d]; e<row_offsets[d+1]; ++e)
            output[sources[e]]+=static_cast<double>(weights[e])*input[d];
}
inline bool near(double actual,double expected,double atol=1e-5,double rtol=1e-5) {
    if (std::isnan(actual)||std::isnan(expected)) return false;
    if (std::isinf(actual)||std::isinf(expected)) return actual==expected;
    return std::abs(actual-expected)<=atol+rtol*std::abs(expected);
}
inline double decode_half(std::uint16_t bits) {
    const int sign=(bits&0x8000)?-1:1;
    const unsigned exponent=(bits>>10)&31u, mantissa=bits&1023u;
    if (exponent==31u) throw std::runtime_error("fixture must be finite");
    if (!exponent) return sign*std::ldexp(static_cast<double>(mantissa),-24);
    return sign*std::ldexp(1.0+mantissa/1024.0,static_cast<int>(exponent)-15);
}
inline void check_fixture() {
    for (std::size_t e=0; e<edges; ++e) {
        if (decode_half(half_weights_1[e])!=weights_1[e] || decode_half(half_weights_2[e])!=weights_2[e])
            throw std::runtime_error("half fixture differs from oracle inputs");
    }
    double y[genes],z[regulators];
    forward_reference(weights_1,state_a,y);
    constexpr double hand_forward[]={0,1.25,0.5,0,1.5};
    transpose_reference(weights_1,gene_signal,z);
    constexpr double hand_transpose[]={0.75,-4.0,0.75,0.125};
    for (std::size_t i=0;i<genes;++i) if(y[i]!=hand_forward[i]) throw std::runtime_error("bad forward fixture");
    for (std::size_t i=0;i<regulators;++i) if(z[i]!=hand_transpose[i]) throw std::runtime_error("bad transpose fixture");
    double left=0,right=0;
    for (std::size_t i=0;i<genes;++i)left+=y[i]*gene_signal[i];
    for (std::size_t i=0;i<regulators;++i)right+=state_a[i]*z[i];
    if (!near(left,right,0,0)) throw std::runtime_error("bad adjoint fixture");
    if (near(std::nan(""),0)) throw std::runtime_error("NaN incorrectly accepted");
}
} // namespace spine_demo
