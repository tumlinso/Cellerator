#include "cpu_kernels.hh"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
#include <immintrin.h>
#endif

namespace velocity::detail::cpu {
namespace {

float half_value(std::uint16_t h) noexcept {
    const auto sign = std::uint32_t(h & 0x8000) << 16;
    auto exponent = unsigned((h >> 10) & 0x1f);
    auto fraction = std::uint32_t(h & 0x3ff);
    if (exponent == 0x1f)
        return std::bit_cast<float>(fraction ? 0x7fc00000u : sign | 0x7f800000u);
    if (!exponent) {
        if (!fraction) return std::bit_cast<float>(sign);
        int e = -14;
        while (!(fraction & 0x400)) { fraction <<= 1; --e; }
        return std::bit_cast<float>(sign | (std::uint32_t(e + 127) << 23) | ((fraction & 0x3ff) << 13));
    }
    return std::bit_cast<float>(sign | (std::uint32_t(int(exponent) + 112) << 23) | (fraction << 13));
}

template<class T> void weighted_add_scalar(const T* source, T weight, T* output, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) output[i] = output[i] + weight*source[i];
}

template<bool SS, bool SU, bool UU, class T>
void products_scalar_run(const T* s, const T* u, T* ss, T* su, T* uu, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (SS) ss[i] = s[i]*s[i];
        if constexpr (SU) su[i] = s[i]*u[i];
        if constexpr (UU) uu[i] = u[i]*u[i];
    }
}

template<class T> void products_scalar(const T* s, const T* u, T* ss, T* su, T* uu,
                                        std::size_t count) noexcept {
    if (ss) {
        if (su) { if (uu) products_scalar_run<true,true,true>(s,u,ss,su,uu,count);
                  else products_scalar_run<true,true,false>(s,u,ss,su,uu,count); }
        else { if (uu) products_scalar_run<true,false,true>(s,u,ss,su,uu,count);
               else products_scalar_run<true,false,false>(s,u,ss,su,uu,count); }
    } else if (su) {
        if (uu) products_scalar_run<false,true,true>(s,u,ss,su,uu,count);
        else products_scalar_run<false,true,false>(s,u,ss,su,uu,count);
    } else if (uu) products_scalar_run<false,false,true>(s,u,ss,su,uu,count);
}

template<class T> void residuals_scalar(const T* s, const T* u, const T* gamma, const T* offset,
                                         T* output, std::size_t rows, std::size_t genes) noexcept {
    for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < genes; ++j)
        output[i*genes + j] = u[i*genes + j] - gamma[j]*s[i*genes + j] - offset[j];
}

template<class T> void column_sums_scalar(const T* values, std::size_t rows, std::size_t genes,
                                          std::ptrdiff_t row_stride, T* output) noexcept {
    for (std::size_t j = 0; j < genes; ++j) {
        T sum = 0;
        for (std::size_t i = 0; i < rows; ++i) sum += values[std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)];
        output[j] = sum;
    }
}

#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
template<class T> void weighted_add_avx2(const T* source, T weight, T* output, std::size_t count) noexcept {
    if (!count) return;
    std::size_t i = 0;
    if constexpr (std::is_same_v<T, float>) {
        const auto w = _mm256_set1_ps(weight);
        for (; i + 8 <= count; i += 8) {
            const auto x = _mm256_loadu_ps(source + i), y = _mm256_loadu_ps(output + i);
            _mm256_storeu_ps(output + i, _mm256_add_ps(y, _mm256_mul_ps(w, x)));
        }
    } else {
        const auto w = _mm256_set1_pd(weight);
        for (; i + 4 <= count; i += 4) {
            const auto x = _mm256_loadu_pd(source + i), y = _mm256_loadu_pd(output + i);
            _mm256_storeu_pd(output + i, _mm256_add_pd(y, _mm256_mul_pd(w, x)));
        }
    }
    weighted_add_scalar(source + i, weight, output + i, count - i);
}

template<bool SS, bool SU, bool UU, class T>
void products_avx2_run(const T* s, const T* u, T* ss, T* su, T* uu, std::size_t count) noexcept {
    std::size_t i = 0;
    if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= count; i += 8) {
            auto x = _mm256_setzero_ps(), y = _mm256_setzero_ps();
            if constexpr (SS || SU) x = _mm256_loadu_ps(s + i);
            if constexpr (SU || UU) y = _mm256_loadu_ps(u + i);
            if constexpr (SS) _mm256_storeu_ps(ss + i, _mm256_mul_ps(x, x));
            if constexpr (SU) _mm256_storeu_ps(su + i, _mm256_mul_ps(x, y));
            if constexpr (UU) _mm256_storeu_ps(uu + i, _mm256_mul_ps(y, y));
        }
    } else {
        for (; i + 4 <= count; i += 4) {
            auto x = _mm256_setzero_pd(), y = _mm256_setzero_pd();
            if constexpr (SS || SU) x = _mm256_loadu_pd(s + i);
            if constexpr (SU || UU) y = _mm256_loadu_pd(u + i);
            if constexpr (SS) _mm256_storeu_pd(ss + i, _mm256_mul_pd(x, x));
            if constexpr (SU) _mm256_storeu_pd(su + i, _mm256_mul_pd(x, y));
            if constexpr (UU) _mm256_storeu_pd(uu + i, _mm256_mul_pd(y, y));
        }
    }
    const T* st = s; const T* ut = u;
    T* sst = ss; T* sut = su; T* uut = uu;
    if constexpr (SS || SU) st += i;
    if constexpr (SU || UU) ut += i;
    if constexpr (SS) sst += i;
    if constexpr (SU) sut += i;
    if constexpr (UU) uut += i;
    products_scalar_run<SS,SU,UU>(st,ut,sst,sut,uut,count-i);
}

template<class T> void products_avx2(const T* s, const T* u, T* ss, T* su, T* uu,
                                      std::size_t count) noexcept {
    if (!count) return;
    if (ss) {
        if (su) { if (uu) products_avx2_run<true,true,true>(s,u,ss,su,uu,count);
                  else products_avx2_run<true,true,false>(s,u,ss,su,uu,count); }
        else { if (uu) products_avx2_run<true,false,true>(s,u,ss,su,uu,count);
               else products_avx2_run<true,false,false>(s,u,ss,su,uu,count); }
    } else if (su) {
        if (uu) products_avx2_run<false,true,true>(s,u,ss,su,uu,count);
        else products_avx2_run<false,true,false>(s,u,ss,su,uu,count);
    } else if (uu) products_avx2_run<false,false,true>(s,u,ss,su,uu,count);
}

template<class T> void residuals_avx2(const T* s, const T* u, const T* gamma, const T* offset,
                                       T* output, std::size_t rows, std::size_t genes) noexcept {
    if (!rows || !genes) return;
    if constexpr (std::is_same_v<T, float>) {
        for (std::size_t i = 0; i < rows; ++i) {
            const auto* sr = s + i*genes; const auto* ur = u + i*genes; auto* out = output + i*genes;
            std::size_t j = 0;
            for (j = 0; j + 8 <= genes; j += 8) {
                const auto x = _mm256_loadu_ps(sr + j), y = _mm256_loadu_ps(ur + j);
                const auto g = _mm256_loadu_ps(gamma + j), a = _mm256_loadu_ps(offset + j);
                _mm256_storeu_ps(out + j, _mm256_sub_ps(_mm256_sub_ps(y, _mm256_mul_ps(g, x)), a));
            }
            for (; j < genes; ++j) out[j] = ur[j] - gamma[j]*sr[j] - offset[j];
        }
    } else {
        for (std::size_t i = 0; i < rows; ++i) {
            const auto* sr = s + i*genes; const auto* ur = u + i*genes; auto* out = output + i*genes;
            std::size_t j = 0;
            for (j = 0; j + 4 <= genes; j += 4) {
                const auto x = _mm256_loadu_pd(sr + j), y = _mm256_loadu_pd(ur + j);
                const auto g = _mm256_loadu_pd(gamma + j), a = _mm256_loadu_pd(offset + j);
                _mm256_storeu_pd(out + j, _mm256_sub_pd(_mm256_sub_pd(y, _mm256_mul_pd(g, x)), a));
            }
            for (; j < genes; ++j) out[j] = ur[j] - gamma[j]*sr[j] - offset[j];
        }
    }
}

template<class T> void column_sums_avx2(const T* values, std::size_t rows, std::size_t genes,
                                         std::ptrdiff_t row_stride, T* output) noexcept {
    std::size_t j = 0;
    if constexpr (std::is_same_v<T, float>) {
        for (; j + 8 <= genes; j += 8) {
            auto sum = _mm256_setzero_ps();
            for (std::size_t i = 0; i < rows; ++i)
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(values + std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)));
            _mm256_storeu_ps(output + j, sum);
        }
    } else {
        for (; j + 4 <= genes; j += 4) {
            auto sum = _mm256_setzero_pd();
            for (std::size_t i = 0; i < rows; ++i)
                sum = _mm256_add_pd(sum, _mm256_loadu_pd(values + std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)));
            _mm256_storeu_pd(output + j, sum);
        }
    }
    for (; j < genes; ++j) {
        T sum = 0;
        for (std::size_t i = 0; i < rows; ++i)
            sum += values[std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)];
        output[j] = sum;
    }
}

void half_to_float_avx2(const std::uint16_t* source, std::ptrdiff_t stride_bytes,
                        float* output, std::size_t count) noexcept {
    auto* p = reinterpret_cast<const unsigned char*>(source);
    std::size_t i = 0;
    const auto exp_mask = _mm_set1_epi16(0x7c00), frac_mask = _mm_set1_epi16(0x03ff);
    const auto inf = _mm_set1_epi16(0x7c00), zero = _mm_setzero_si128();
    const auto qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
    for (; i + 8 <= count; i += 8) {
        __m128i h;
        if (stride_bytes == 2) h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        else {
            std::uint16_t lane[8];
            for (std::size_t k = 0; k < 8; ++k) std::memcpy(lane + k, p + std::ptrdiff_t(k)*stride_bytes, 2);
            h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lane));
        }
        const auto e = _mm_and_si128(h, exp_mask), f = _mm_and_si128(h, frac_mask);
        const auto nan = _mm_andnot_si128(_mm_cmpeq_epi16(f, zero), _mm_cmpeq_epi16(e, inf));
        const auto finite = _mm_andnot_si128(nan, h);
        const auto x = _mm256_cvtph_ps(finite);
        const auto mask = _mm256_castsi256_ps(_mm256_cvtepi16_epi32(nan));
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(x, qnan, mask));
        p += std::ptrdiff_t(8)*stride_bytes;
    }
    for (; i < count; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p, 2);
        output[i] = half_value(h);
        p += stride_bytes;
    }
}
#endif

}

bool avx2_enabled() noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    return true;
#else
    return false;
#endif
}

void half_to_float(const std::uint16_t* source, std::ptrdiff_t stride_bytes,
                   float* output, std::size_t count) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    half_to_float_avx2(source, stride_bytes, output, count);
#else
    auto* p = reinterpret_cast<const unsigned char*>(source);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p, 2);
        output[i] = half_value(h);
        p += stride_bytes;
    }
#endif
}

void weighted_add(const float* source, float weight, float* output, std::size_t count) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    weighted_add_avx2(source, weight, output, count);
#else
    weighted_add_scalar(source, weight, output, count);
#endif
}
void weighted_add(const double* source, double weight, double* output, std::size_t count) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    weighted_add_avx2(source, weight, output, count);
#else
    weighted_add_scalar(source, weight, output, count);
#endif
}

void products(const float* s, const float* u, float* ss, float* su, float* uu, std::size_t count) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    products_avx2(s, u, ss, su, uu, count);
#else
    products_scalar(s, u, ss, su, uu, count);
#endif
}
void products(const double* s, const double* u, double* ss, double* su, double* uu, std::size_t count) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    products_avx2(s, u, ss, su, uu, count);
#else
    products_scalar(s, u, ss, su, uu, count);
#endif
}

void residuals(const float* s, const float* u, const float* gamma, const float* offset,
               float* output, std::size_t rows, std::size_t genes) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    residuals_avx2(s, u, gamma, offset, output, rows, genes);
#else
    residuals_scalar(s, u, gamma, offset, output, rows, genes);
#endif
}
void residuals(const double* s, const double* u, const double* gamma, const double* offset,
               double* output, std::size_t rows, std::size_t genes) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    residuals_avx2(s, u, gamma, offset, output, rows, genes);
#else
    residuals_scalar(s, u, gamma, offset, output, rows, genes);
#endif
}

void column_sums(const float* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, float* output) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    column_sums_avx2(values, rows, genes, row_stride, output);
#else
    column_sums_scalar(values, rows, genes, row_stride, output);
#endif
}
void column_sums(const double* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, double* output) noexcept {
#if defined(VELOCITY_AVX2) && VELOCITY_AVX2
    column_sums_avx2(values, rows, genes, row_stride, output);
#else
    column_sums_scalar(values, rows, genes, row_stride, output);
#endif
}

}
