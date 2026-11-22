#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

import ccell.velocity;

#include "stochastic_support.hh"

namespace {
namespace velocity = ccell::velocity;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Independent finite binary16 decoder for the FP32 oracle.
template<size_t N>
std::array<float, N> widen(const uint16_t (&bits)[N]) {
    std::array<float, N> values{};
    for (size_t i = 0; i < N; ++i) {
        const unsigned exponent = (bits[i] >> 10) & 31;
        const unsigned fraction = bits[i] & 1023;
        const float value = exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                                     : std::ldexp(float(fraction), -24);
        values[i] = bits[i] & 0x8000 ? -value : value;
    }
    return values;
}

auto half_dense(const uint16_t* values, size_t n, size_t g, ptrdiff_t row, ptrdiff_t col = 1) {
    return velocity::borrow_binary16(cellerator::memory::const_matrix_view<uint16_t, ptrdiff_t>{
        values, n, g, row, {}, col});
}

template<class Index, class Offset>
auto half_sparse(SparseAxis axis, size_t n, size_t g, size_t nnz, const uint16_t* values,
                 const Index* indices, const Offset* offsets) {
    return velocity::borrow_binary16(
        cellerator::tensor::base::CompressedView<Index, const uint16_t, Offset>{
            {n, g, nnz, axis, offsets, indices}, values});
}

void compare(const PreparedStorage& source, const PreparedStorage& oracle) {
    const auto& p = source.value;
    const auto& q = oracle.value;
    const size_t n = p.rows * p.genes;
    std::vector<float> ms(n), mu(n), es(n), eu(n);
    velocity::reference::moments(p, 0, p.genes, ms.data(), mu.data());
    velocity::reference::moments(q, 0, p.genes, es.data(), eu.data());
    check(ms == es && mu == eu, "FP16 first moments match widened FP32");

    std::vector<float> mss(n), mus(n), muu(n), ess(n), eus(n), euu(n), src_s(p.rows), src_u(p.rows);
    velocity::reference::second_moments(p, 0, p.genes, ms.data(), mu.data(), mss.data(),
        mus.data(), muu.data(), src_s.data(), src_u.data());
    velocity::reference::second_moments(q, 0, p.genes, es.data(), eu.data(), ess.data(),
        eus.data(), euu.data(), src_s.data(), src_u.data());
    check(ms == es && mu == eu && mss == ess && mus == eus && muu == euu,
          "FP16 second moments match widened FP32");

    velocity::StochasticOptions opts;
    opts.initialization.trimming = velocity::Trimming::none;
    opts.initialization.fit_intercept = true;
    opts.initialization.min_r2 = 0;
    opts.initialization.min_ratio = 0;
    StochasticStorage<float> fits(p), expected(q);
    velocity::WorkspaceSize size;
    require_status(velocity::workspace_size<float>(p, {}, false, size));
    std::vector<float> scratch(size.scalars);
    std::vector<uint8_t> masks(size.masks);
    velocity::Workspace<float> workspace;
    require_status(velocity::bind_workspace(p, {}, false, scratch.data(), scratch.size(),
                                          masks.data(), masks.size(), workspace));
    std::vector<float> out(n), ref(n);
    require_status(velocity::reference::steady_state_into(p, workspace, fits.result.initialization,
        out.data(), n, p.genes, opts.initialization));
    require_status(velocity::bind_workspace(q, {}, false, scratch.data(), scratch.size(),
                                          masks.data(), masks.size(), workspace));
    require_status(velocity::reference::steady_state_into(q, workspace, expected.result.initialization,
        ref.data(), n, q.genes, opts.initialization));
    check(out == ref && fits.ig == expected.ig && fits.io == expected.io &&
          fits.ir == expected.ir && fits.initial_status == expected.initial_status,
          "complete FP16 steady-state fit and residuals match widened FP32");

    std::vector<float> second(n), expected_second(n);
    require_status(velocity::reference::stochastic_into(p, fits.workspace, fits.result,
        out.data(), n, second.data(), n, p.genes, opts));
    require_status(velocity::reference::stochastic_into(q, expected.workspace, expected.result,
        ref.data(), n, expected_second.data(), n, q.genes, opts));
    check(out == ref && second == expected_second && fits.ig == expected.ig &&
          fits.gamma == expected.gamma && fits.offset == expected.offset &&
          fits.offset2 == expected.offset2 && fits.first == expected.first &&
          fits.second == expected.second && fits.objective == expected.objective &&
          fits.status == expected.status && fits.result.refit_all == expected.result.refit_all,
          "complete FP16 stochastic fit and residuals match widened FP32");
    for (size_t gene = 0; gene < p.genes; ++gene)
        check(fits.selected[gene] == expected.selected[gene], "FP16 preserves gene selection");
    check(velocity::workspace_size<double>(p, {}, false, size).code == velocity::Code::dtype,
          "binary16 inputs require FP32 workspace arithmetic");
}

void dense() {
    constexpr size_t n = 4, g = 2;
    uint16_t s[]{0x4000, 0x3c00, 0x4200, 0x0001, 0x4400, 0x4000, 0x4500, 0x4200};
    const uint16_t u[]{0x4300, 0x4000, 0x4480, 0x3c00, 0x4600, 0x4400, 0x4840, 0x4600};
    const uint16_t w[]{0x3a00, 0x3400, 0, 0, 0, 0x3a00, 0x3400, 0,
                       0, 0, 0x3a00, 0x3400, 0x3400, 0, 0, 0x3a00};
    auto hs = half_dense(s + n * g - 1, n, g, -ptrdiff_t(g), -1);
    auto hu = half_dense(u + n * g - 1, n, g, -ptrdiff_t(g), -1);
    auto hw = half_dense(w, n, n, n);
    PreparedStorage source(hs, hu, hw);
    s[0] = 0x4100;
    const auto fs = widen(s), fu = widen(u);
    const auto fw = widen(w);
    PreparedStorage oracle(borrow_dense(fs.data() + n * g - 1, {n, g}, {-ptrdiff_t(g), -1}),
                           borrow_dense(fu.data() + n * g - 1, {n, g}, {-ptrdiff_t(g), -1}),
                           borrow_dense(fw.data(), {n, n}, {n, 1}));
    compare(source, oracle);

    s[0] = 0x7c00;
    check(velocity::validate({hs, hu, hw}).code == velocity::Code::nonfinite,
          "binary16 infinity is rejected during dense validation");
    const auto integer = borrow_dense(s, {n, g}, {g, 1});
    check(!velocity::validate({integer, hu, hw}), "untagged uint16 is not a floating input");
}

void sparse() {
    constexpr size_t n = 4, g = 1;
    const uint16_t s[]{0x3c00, 0x4000, 0x4000, 0x4400, 0x4500};
    uint16_t u[]{0x4200, 0x4000, 0x4200, 0x4600, 0x4800};
    const uint16_t w[]{0x3c00, 0x3800, 0x3800, 0x3c00, 0x3c00};
    const int16_t si[]{0, 0, 0, 0, 0};
    const uint32_t ui[]{0, 0, 1, 2, 3}, wi[]{0, 1, 1, 2, 3};
    const int64_t so[]{0, 2, 3, 4, 5}, wo[]{0, 1, 3, 4, 5};
    const uint64_t uo[]{0, 5};
    const auto hs = half_sparse(SparseAxis::row, n, g, 5, s, si, so);
    const auto hu = half_sparse(SparseAxis::column, n, g, 5, u, ui, uo);
    const auto hw = half_sparse(SparseAxis::column, n, n, 5, w, wi, wo);
    PreparedStorage source(hs, hu, hw);
    const auto fs = widen(s), fu = widen(u), fw = widen(w);
    PreparedStorage oracle(borrow_sparse(SparseAxis::row, n, g, 5, fs.data(), si, so),
                           borrow_sparse(SparseAxis::column, n, g, 5, fu.data(), ui, uo),
                           borrow_sparse(SparseAxis::column, n, n, 5, fw.data(), wi, wo));
    compare(source, oracle);
    float mss[n]{}, mus[n]{}, muu[n]{}, src_s[n]{}, src_u[n]{};
    velocity::reference::second_moments<float>(source.value, 0, g, nullptr, nullptr,
        mss, mus, muu, src_s, src_u);
    check(mss[0] == 9 && mus[0] == 15 && muu[0] == 25,
          "binary16 sparse duplicates combine before FP32 products");

    velocity::reference::NeighborhoodSize neighborhood;
    require_status(velocity::reference::neighborhood_size(hw, neighborhood));
    std::vector<size_t> indices(neighborhood.indices), columns(neighborhood.entries);
    std::vector<velocity::Entry> entries(neighborhood.scratch_entries);
    std::vector<float> weights(neighborhood.entries);
    size_t offsets[n + 1]{}, used = 0;
    require_status(velocity::reference::prepare_neighborhood(hw,
        {indices.data(), indices.size(), entries.data(), entries.size(), weights.data(),
         columns.data(), offsets, columns.size(), n + 1}, used));
    check(used == n && offsets[n] == n, "binary16 neighborhood coalesces duplicate self edges");
    for (size_t row = 0; row < n; ++row)
        check(offsets[row] == row && columns[row] == row && weights[row] == 1,
              "binary16 neighborhood emits normalized FP32 weights");

    u[2] = 0x7e01;
    check(velocity::validate({hs, hu, hw}).code == velocity::Code::nonfinite,
          "binary16 NaN is rejected during sparse validation");
}
}

int main() {
    dense();
    sparse();
}
