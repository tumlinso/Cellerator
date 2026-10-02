#include <cmath>
#include <cstdint>
#include <algorithm>

namespace {
int admit(std::int64_t n, std::int64_t m, const float* x, const float* k,
          const std::int64_t* a, const std::int64_t* b) {
    if (n < 0 || m < 0 || (n && !x) || (m && (!k || !a || !b))) return 1;
    for (std::int64_t i = 0; i < m; ++i)
        if (a[i] < 0 || a[i] >= n || b[i] < 0 || b[i] >= n) return 2;
    return 0;
}
}
extern "C" {
int product_forward(std::int64_t n, std::int64_t m, const float* x, const float* k,
                    const std::int64_t* a, const std::int64_t* b, float* y) {
    const int error = admit(n, m, x, k, a, b);
    if (error) return error;
    if (m && !y) return 1;
    for (std::int64_t i = 0; i < m; ++i) y[i] = (k[i] * x[a[i]]) * x[b[i]];
    return 0;
}
int product_vjp(std::int64_t n, std::int64_t m, const float* x, const float* k,
                const std::int64_t* a, const std::int64_t* b, const float* g,
                float* dx, float* dk) {
    const int error = admit(n, m, x, k, a, b);
    if (error) return error;
    if ((n && !dx) || (m && (!g || !dk))) return 1;
    if (n) std::fill(dx, dx + n, 0.0f);
    for (std::int64_t i = 0; i < m; ++i) {
        dx[a[i]] += (g[i] * k[i]) * x[b[i]];
        dx[b[i]] += (g[i] * k[i]) * x[a[i]];
        dk[i] = (g[i] * x[a[i]]) * x[b[i]];
    }
    return 0;
}
int product_jvp(std::int64_t n, std::int64_t m, const float* x, const float* k,
                const std::int64_t* a, const std::int64_t* b, const float* dx,
                const float* dk, float* dy) {
    const int error = admit(n, m, x, k, a, b);
    if (error) return error;
    if ((n && !dx) || (m && (!dk || !dy))) return 1;
    for (std::int64_t i = 0; i < m; ++i) {
        const float state = k[i] * std::fma(dx[a[i]], x[b[i]], x[a[i]] * dx[b[i]]);
        dy[i] = state + (dk[i] * x[a[i]]) * x[b[i]];
    }
    return 0;
}
}
