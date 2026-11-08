#include <anndata/device_abi.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

import velocity.cuda;

extern "C" std::uint64_t bio_device_rows(anndata::device::DenseView);
static_assert(std::is_same_v<anndata::cuda::DenseView, anndata::device::DenseView>);

int main() try {
    namespace ac = anndata::cuda;
    namespace vc = velocity::cuda;
    using ac::ScalarType;
    constexpr std::size_t n = 4, g = 2;
    ac::Stream stream{nullptr, 0};
    auto upload = [&](const auto& data, std::size_t rows, std::size_t cols) {
        ac::DeviceBuffer storage(sizeof(data), 0);
        ac::copy_to_device(storage.span(), 0, data.data(), sizeof(data), stream);
        ac::DenseView view{storage.span(), 0, rows, cols, std::int64_t(cols), 1, ScalarType::float32};
        return std::pair{std::move(storage), view};
    };
    std::array<float, n*g> s{1, 2, 2, 4, 3, 6, 4, 8}, u{};
    std::array<float, n*n> weights{};
    for (std::size_t i = 0; i < s.size(); ++i) u[i] = 2*s[i];
    for (std::size_t i = 0; i < n; ++i) weights[i*n+i] = 1;
    auto [ds, sv] = upload(s, n, g);
    auto [du, uv] = upload(u, n, g);
    auto [dw, wv] = upload(weights, n, n);
    if (bio_device_rows(sv) != n) throw std::runtime_error("NVCC/module ABI mismatch");
    vc::PreparedInput input({wv, sv, uv}, stream);
    velocity::RunOptions run;
    run.compute_precision = velocity::ComputePrecision::float32;
    vc::Workspace scratch(input, run);
    ac::DeviceBuffer output(sizeof(s), 0);
    vc::Outputs destinations;
    destinations.velocity = {output.span(), 0, n, g, g, 1, ScalarType::float32};
    velocity::FitOptions fit;
    fit.trimming = velocity::Trimming::none;
    auto result = vc::reference::steady_state(input, scratch, fit, destinations, stream);
    std::array<float, n*g> velocity_values{};
    std::array<float, g> gamma{};
    ac::copy_to_host(velocity_values.data(), output.span(), 0, sizeof(velocity_values), stream);
    const auto coefficients = result.view().initial_gamma;
    ac::copy_to_host(gamma.data(), coefficients.values, coefficients.origin, sizeof(gamma), stream);
    ac::synchronize(stream);
    for (auto x : velocity_values) if (std::abs(x) > 2e-5f) throw std::runtime_error("nonzero residual");
    for (auto x : gamma) if (std::abs(x-2) > 2e-5f) throw std::runtime_error("unexpected fitted gamma");
    std::cout << "installed CUDA consumer passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
