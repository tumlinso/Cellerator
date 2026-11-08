#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include <Cellerator/ops/numeric/binary16.hh>

import velocity.precision;

namespace {
std::atomic<bool> watch_allocations = false;
std::atomic<std::size_t> allocation_count = 0;
}

#if defined(__clang__)
#if __has_feature(thread_sanitizer)
#define VELOCITY_TSAN_TEST
#endif
#elif defined(__SANITIZE_THREAD__)
#define VELOCITY_TSAN_TEST
#endif

// TSan supplies strong new/delete interceptors; allocation checks run in the
// ordinary and ASan/UBSan suites instead.
#ifndef VELOCITY_TSAN_TEST
void* operator new(std::size_t n) {
    if (watch_allocations.load(std::memory_order_relaxed)) allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    if (watch_allocations.load(std::memory_order_relaxed)) allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, std::align_val_t alignment) {
    if (watch_allocations.load(std::memory_order_relaxed)) allocation_count.fetch_add(1, std::memory_order_relaxed);
    auto a = std::size_t(alignment), size = n ? n : 1;
    size = ((size + a - 1) / a) * a;
    if (auto* p = std::aligned_alloc(a, size)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t alignment) {
    return ::operator new(n, alignment);
}
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

#endif

namespace {
int failures = 0;
void expect(bool ok, const char* message) {
    if (ok) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}
bool close(double a, double b, double tol = 2e-10) {
    return std::abs(a - b) <= tol * (1 + std::abs(b));
}

constexpr std::size_t n = 8, source_genes = 5, genes = 7;
constexpr std::array<velocity::GeneRange, 3> ranges{{{4, 3}, {2, 2}, {0, 2}}};

struct Fixture {
    std::vector<double> s = [] {
        std::vector<double> x(n * source_genes);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < source_genes; ++j)
                x[i * source_genes + j] = 1 + double(i) + .2 * double(j);
        return x;
    }();
    std::vector<double> u = [&] {
        std::vector<double> x(n * source_genes);
        constexpr double noise[source_genes]{0, .015, .035, .07, .12};
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < source_genes; ++j) {
                const auto k = i * source_genes + j;
                x[k] = (1.25 + .05 * double(j)) * s[k] + noise[j] * (i % 2 ? 1 : -1);
            }
        x[6 * source_genes + 1] += 9; // This row is excluded from fitting.
        return x;
    }();
    std::vector<double> w = [] {
        std::vector<double> x(n * n);
        for (std::size_t i = 0; i < n; ++i) x[i * n + i] = 1;
        return x;
    }();
    std::array<bool, n> fit_rows{true, true, true, true, true, true, false, true};
    std::array<bool, source_genes> allowed{true, true, true, true, false};
    anndatapp::Array sa = anndatapp::Array::borrow_dense(s.data(), {n, source_genes},
        {std::ptrdiff_t(source_genes), 1});
    anndatapp::Array ua = anndatapp::Array::borrow_dense(u.data(), {n, source_genes},
        {std::ptrdiff_t(source_genes), 1});
    anndatapp::Array wa = anndatapp::Array::borrow_dense(w.data(), {n, n},
        {std::ptrdiff_t(n), 1});
    anndatapp::Array fa = anndatapp::Array::borrow_dense(fit_rows.data(), {n}, {1});
    anndatapp::Array ga = anndatapp::Array::borrow_dense(allowed.data(), {source_genes}, {1});
    velocity::PreparedInput p = [&] {
        velocity::Input input{sa.view(), ua.view(), wa.view()};
        input.genes = anndatapp::Axis::gather({4, 1, 4, 0, 3, 2, 1});
        input.fit_mask = fa.view();
        input.gene_mask = ga.view();
        return velocity::PreparedInput(std::move(input));
    }();
};

velocity::FitOptions fit_options() {
    velocity::FitOptions o;
    o.trimming = velocity::Trimming::none;
    o.fit_intercept = true;
    o.positive_intercept = false;
    o.min_r2 = 0;
    o.min_ratio = 0;
    return o;
}
velocity::RunOptions run_options() {
    velocity::RunOptions o;
    o.block_genes = 3;
    o.output_dtype = velocity::StorageType::float64;
    return o;
}

template<class T> bool same_fit(const velocity::FitResult& a, const velocity::FitResult& b) {
    auto ag = a.gamma.view().dense<T>(), bg = b.gamma.view().dense<T>();
    auto ao = a.offset.view().dense<T>(), bo = b.offset.view().dense<T>();
    auto ar = a.r2.view().dense<T>(), br = b.r2.view().dense<T>();
    auto as = a.genes.view().dense<bool>(), bs = b.genes.view().dense<bool>();
    auto at = a.status.view().dense<std::uint8_t>(), bt = b.status.view().dense<std::uint8_t>();
    bool ok = a.low_signal == b.low_signal;
    for (std::size_t j = 0; j < genes; ++j)
        ok = ok && close(ag.data()[j], bg.data()[j]) && close(ao.data()[j], bo.data()[j]) &&
            close(ar.data()[j], br.data()[j]) && as.data()[j] == bs.data()[j] &&
            at.data()[j] == bt.data()[j];
    return ok;
}

template<class T> bool same_matrix(const anndatapp::Array& a, const anndatapp::Array& b,
                                   std::size_t rows, std::size_t columns) {
    auto av = a.view().dense<T>(), bv = b.view().dense<T>();
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j) {
            auto x = av.data()[std::ptrdiff_t(i) * av.stride(0) + std::ptrdiff_t(j) * av.stride(1)];
            auto y = bv.data()[std::ptrdiff_t(i) * bv.stride(0) + std::ptrdiff_t(j) * bv.stride(1)];
            if (!close(x, y)) return false;
        }
    return true;
}

void scheduled_range(const velocity::PreparedInput& p, velocity::FitBuffers<double> buffers,
                     velocity::DenseOutput output, velocity::GeneRange range,
                     velocity::FitOptions options) {
    auto run = run_options();
    velocity::Workspace workspace(p, run);
    auto m = velocity::reference::moments<double>(p, workspace, range);
    auto f = velocity::reference::fit(p, workspace, m, buffers, options);
    velocity::reference::write_residuals(m, f, output);
}

void deterministic_blocks_match_owner_and_masks() {
    Fixture x;
    auto options = fit_options();
    auto serial_output = anndatapp::Array::zeros<double>({n, genes});
    velocity::Workspace serial_workspace(x.p, run_options());
    auto serial = velocity::reference::steady_state_into(x.p, serial_workspace,
        serial_output.edit(), options);

    auto result = velocity::make_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    auto scheduled_output = anndatapp::Array::zeros<double>({n, genes});
    auto output = velocity::acquire_output(x.p, scheduled_output.edit());
    velocity::validate_run(x.p);
    velocity::Workspace workspace(x.p, run_options());
    for (auto range : ranges) {
        auto m = velocity::reference::moments<double>(x.p, workspace, range);
        auto f = velocity::reference::fit(x.p, workspace, m, buffers, options);
        velocity::reference::write_residuals(m, f, output);
    }
    std::array<double, genes> score{};
    result.low_signal = velocity::reference::finalize_selection(
        x.p, buffers, options, score.data(), score.size());
    expect(same_fit<double>(result, serial),
           "reverse-order disjoint ranges match owning diagnostics and global fit-mask selection");
    expect(same_matrix<double>(scheduled_output, serial_output, n, genes),
           "caller-scheduled residual blocks write the same global output columns as the owner");

    constexpr std::array<std::size_t, genes> source_order{4, 1, 4, 0, 3, 2, 1};
    bool masks_ok = true;
    auto gamma = result.gamma.view().dense<double>();
    auto selected = result.genes.view().dense<bool>();
    double mx = 0, my = 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) if (x.fit_rows[i]) {
        mx += x.s[i * source_genes + 1];
        my += x.u[i * source_genes + 1];
        ++count;
    }
    mx /= double(count); my /= double(count);
    double xx = 0, xy = 0;
    for (std::size_t i = 0; i < n; ++i) if (x.fit_rows[i]) {
        const double dx = x.s[i * source_genes + 1] - mx;
        xx += dx * dx;
        xy += dx * (x.u[i * source_genes + 1] - my);
    }
    const double expected_gamma = xy / xx;
    for (std::size_t j = 0; j < genes; ++j) {
        masks_ok = masks_ok && x.p.gene_index(j) == source_order[j];
        if (source_order[j] == 4) masks_ok = masks_ok && !selected.data()[j];
        if (source_order[j] == 1)
            masks_ok = masks_ok && selected.data()[j] && close(gamma.data()[j], expected_gamma);
    }
    expect(masks_ok, "duplicate gene gathers retain order, gene masks exclude both copies, and row masks exclude the outlier");
}

void real_threads_match_owner_and_do_not_allocate_after_setup() {
    Fixture x;
    auto options = fit_options();
    auto serial_output = anndatapp::Array::zeros<double>({n, genes});
    velocity::Workspace serial_workspace(x.p, run_options());
    auto serial = velocity::reference::steady_state_into(x.p, serial_workspace,
        serial_output.edit(), options);

    auto result = velocity::make_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    auto threaded_output = anndatapp::Array::zeros<double>({n, genes});
    auto output = velocity::acquire_output(x.p, threaded_output.edit());
    velocity::validate_run(x.p);
    std::array<std::exception_ptr, ranges.size()> errors{};
    std::array<std::thread, ranges.size()> workers;
    for (std::size_t i = 0; i < ranges.size(); ++i)
        workers[i] = std::thread([&, i] {
            try { scheduled_range(x.p, buffers, output, ranges[i], options); }
            catch (...) { errors[i] = std::current_exception(); }
        });
    for (auto& worker : workers) worker.join();
    bool threads_ok = true;
    for (auto error : errors) threads_ok = threads_ok && !error;
    std::array<double, genes> score{};
    result.low_signal = velocity::reference::finalize_selection(
        x.p, buffers, options, score.data(), score.size());
    expect(threads_ok && same_fit<double>(result, serial),
           "workers with distinct workspaces and disjoint diagnostics match the serial owner");
    expect(same_matrix<double>(threaded_output, serial_output, n, genes),
           "workers write disjoint output gene columns without changing residual values");

    auto noalloc_run = run_options();
    noalloc_run.block_genes = genes;
    velocity::Workspace workspace(x.p, noalloc_run);
    auto noalloc_result = velocity::make_fit_result<double>(genes);
    auto noalloc_buffers = velocity::fit_buffers<double>(noalloc_result);
    auto noalloc_output = anndatapp::Array::zeros<double>({n, genes});
    auto noalloc_dense = velocity::acquire_output(x.p, noalloc_output.edit());
    velocity::validate_run(x.p);
    allocation_count.store(0, std::memory_order_relaxed);
    watch_allocations.store(true, std::memory_order_relaxed);
    try {
        auto m = velocity::reference::moments<double>(x.p, workspace, {0, genes});
        auto f = velocity::reference::fit(x.p, workspace, m, noalloc_buffers, options);
        velocity::reference::write_residuals(m, f, noalloc_dense);
        noalloc_result.low_signal = velocity::reference::finalize_selection(
            x.p, noalloc_buffers, options, score.data(), score.size());
    } catch (...) {
        watch_allocations.store(false, std::memory_order_relaxed);
        throw;
    }
    watch_allocations.store(false, std::memory_order_relaxed);
    expect(allocation_count.load(std::memory_order_relaxed) == 0,
           "prepared deterministic block calls allocate no C++ heap storage");
}

void low_signal_recovery_runs_after_every_range() {
    Fixture x;
    auto options = fit_options();
    options.min_r2 = 1.01;
    options.low_signal = velocity::LowSignal::scvelo_percentile;
    auto result = velocity::make_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    velocity::Workspace workspace(x.p, run_options());
    velocity::validate_run(x.p);
    for (auto range : ranges) {
        auto m = velocity::reference::moments<double>(x.p, workspace, range);
        velocity::reference::fit(x.p, workspace, m, buffers, options);
    }
    buffers.selected[0] = true; // Gene mask excludes both occurrences.
    buffers.selected[2] = true;
    buffers.selected[1] = true;
    buffers.status[1] = std::uint8_t(velocity::FitStatus::singular);
    std::array<double, genes> score{};
    result.low_signal = velocity::reference::finalize_selection(
        x.p, buffers, options, score.data(), score.size());
    auto selected = result.genes.view().dense<bool>();
    std::size_t count = 0;
    for (std::size_t j = 0; j < genes; ++j) count += selected.data()[j];
    expect(result.low_signal && count == 1 && selected.data()[3] &&
           !selected.data()[0] && !selected.data()[1] && !selected.data()[2],
           "global recovery ignores selected flags on masked or invalid fits and sees all disjoint ranges");
}

void stochastic_initialization_and_refit_barriers_match_owner() {
    Fixture x;
    velocity::StochasticOptions options;
    options.initialization = fit_options();
    options.initialization.min_r2 = 2;
    options.refit_all_if_none = true;
    velocity::StochasticRunOptions run;
    run.block_genes = 3;
    run.output_dtype = velocity::StorageType::float64;
    run.second_residual = true;
    run.spliced_second = true;
    run.cross = true;

    auto serial_velocity = anndatapp::Array::zeros<double>({n, genes});
    auto serial_second = anndatapp::Array::zeros<double>({n, genes});
    velocity::StochasticWorkspace serial_workspace(x.p, run);
    auto serial = velocity::reference::stochastic_into(x.p, serial_workspace,
        serial_velocity.edit(), options, serial_second.edit());

    auto result = velocity::make_stochastic_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    std::array<double, genes> score{};
    velocity::validate_run(x.p);
    std::array<std::exception_ptr, ranges.size()> errors{};
    std::array<std::thread, ranges.size()> initial_workers;
    for (std::size_t i = 0; i < ranges.size(); ++i)
        initial_workers[i] = std::thread([&, i] {
            try {
                velocity::StochasticWorkspace workspace(x.p, run);
                auto m = velocity::reference::initial_moments<double>(x.p, workspace, ranges[i]);
                velocity::reference::fit(x.p, workspace, m, buffers.initialization, options.initialization);
            } catch (...) { errors[i] = std::current_exception(); }
        });
    for (auto& worker : initial_workers) worker.join();
    bool threads_ok = true;
    for (auto error : errors) threads_ok = threads_ok && !error;
    const auto selection = velocity::reference::finalize_selection(
        x.p, buffers, options, score.data(), score.size());
    result.refit_all = selection.refit_all;
    auto velocity_output = anndatapp::Array::zeros<double>({n, genes});
    auto second_output = anndatapp::Array::zeros<double>({n, genes});
    auto velocity_dense = velocity::acquire_output(x.p, velocity_output.edit());
    auto second_dense = velocity::acquire_output(x.p, second_output.edit());
    errors = {};
    std::array<std::thread, ranges.size()> refinement_workers;
    for (std::size_t i = 0; i < ranges.size(); ++i)
        refinement_workers[i] = std::thread([&, i] {
            try {
                velocity::StochasticWorkspace workspace(x.p, run);
                auto m = velocity::reference::second_moments<double>(x.p, workspace, ranges[i]);
                auto scales = velocity::reference::scales(x.p, workspace, m, buffers, options, selection);
                auto f = velocity::reference::fit(x.p, workspace, m, scales, buffers, options, selection);
                velocity::reference::write_residuals(m, f, velocity_dense, second_dense);
            } catch (...) { errors[i] = std::current_exception(); }
        });
    for (auto& worker : refinement_workers) worker.join();
    for (auto error : errors) threads_ok = threads_ok && !error;
    expect(selection.refit_all && threads_ok && result.refit_all == serial.refit_all,
           "stochastic refit-all is decided once between complete initialization and refinement passes");
    expect(same_fit<double>(result.initialization, serial.initialization),
           "stochastic initialization diagnostics match the owning serial result");
    bool diag_ok = true;
    auto gamma = result.gamma.view().dense<double>(), sg = serial.gamma.view().dense<double>();
    auto offset = result.offset.view().dense<double>(), so = serial.offset.view().dense<double>();
    auto offset2 = result.offset2.view().dense<double>(), so2 = serial.offset2.view().dense<double>();
    auto scale1 = result.first_scale.view().dense<double>(), ss1 = serial.first_scale.view().dense<double>();
    auto scale2 = result.second_scale.view().dense<double>(), ss2 = serial.second_scale.view().dense<double>();
    auto status = result.status.view().dense<std::uint8_t>(), ss = serial.status.view().dense<std::uint8_t>();
    auto objective = result.objective.view().dense<double>(), sj = serial.objective.view().dense<double>();
    for (std::size_t j = 0; j < genes; ++j)
        diag_ok = diag_ok && close(gamma.data()[j], sg.data()[j]) && close(offset.data()[j], so.data()[j]) &&
            close(offset2.data()[j], so2.data()[j]) && close(scale1.data()[j], ss1.data()[j]) &&
            close(scale2.data()[j], ss2.data()[j]) && status.data()[j] == ss.data()[j] &&
            close(objective.data()[j], sj.data()[j]);
    expect(diag_ok, "stochastic global diagnostics match the owning serial result");
    expect(same_matrix<double>(velocity_output, serial_velocity, n, genes) &&
           same_matrix<double>(second_output, serial_second, n, genes),
           "stochastic worker blocks write matching velocity and second-residual outputs");
}

void stochastic_low_signal_selection_waits_for_all_ranges() {
    Fixture x;
    velocity::StochasticOptions options;
    options.initialization = fit_options();
    options.initialization.min_r2 = 1.01;
    options.initialization.low_signal = velocity::LowSignal::scvelo_percentile;
    auto result = velocity::make_stochastic_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    velocity::StochasticRunOptions run;
    run.block_genes = 3;
    velocity::validate_run(x.p);
    for (auto range : ranges) {
        velocity::StochasticWorkspace workspace(x.p, run);
        auto m = velocity::reference::initial_moments<double>(x.p, workspace, range);
        velocity::reference::fit(x.p, workspace, m, buffers.initialization, options.initialization);
    }
    std::array<double, genes> score{};
    auto selection = velocity::reference::finalize_selection(
        x.p, buffers, options, score.data(), score.size());
    auto selected = result.initialization.genes.view().dense<bool>();
    std::size_t count = 0;
    for (std::size_t j = 0; j < genes; ++j) count += selected.data()[j];
    expect(selection.low_signal && !selection.refit_all && count == 1 && selected.data()[3],
           "stochastic low-signal recovery selects from the complete initialization population");
}

struct OriginalColumns {
    std::array<float, 4> s{10, 20, 30, 40}, u{50, 60, 70, 80};
    bool available = false;
    std::size_t calls = 0;
    static bool acquire(void* context, std::size_t gene, velocity::precision::OriginalColumn& column) {
        auto& self = *static_cast<OriginalColumns*>(context);
        ++self.calls;
        if (gene || !self.available) return false;
        auto s = anndatapp::Array::borrow_dense(self.s.data(), {self.s.size(), 1}, {1, 1});
        auto u = anndatapp::Array::borrow_dense(self.u.data(), {self.u.size(), 1}, {1, 1});
        column = {s.view(), u.view()};
        return true;
    }
    velocity::precision::OriginalSource source() { return {this, acquire, 0}; }
};

void unavailable_initial_promotion_stays_frozen_for_refinement() {
    constexpr std::size_t rows = 4;
    std::array<std::uint16_t, rows> hs{}, hu{};
    std::array<float, rows * rows> weights{};
    for (std::size_t i = 0; i < rows; ++i) {
        hs[i] = cellerator::numeric::round_f16(float(i + 1));
        hu[i] = cellerator::numeric::round_f16(2 * float(i + 1));
        weights[i * rows + i] = 1;
    }
    auto s = velocity::NumericView::borrow_dense_f16(hs.data(), {rows, 1}, {1, 1});
    auto u = velocity::NumericView::borrow_dense_f16(hu.data(), {rows, 1}, {1, 1});
    auto w = anndatapp::Array::borrow_dense(weights.data(), {rows, rows}, {std::ptrdiff_t(rows), 1});
    velocity::PreparedInput p({s, u, w.view()});
    velocity::StochasticRunOptions run;
    run.block_genes = 1;
    run.compute_precision = velocity::ComputePrecision::float32;
    velocity::StochasticWorkspace workspace(p, run);
    auto initial = velocity::reference::initial_moments<float>(p, workspace, {0, 1});
    OriginalColumns source;
    const std::uint8_t requested[]{1};
    velocity::precision::PromotionOptions options{requested, 1, 1u << 20};
    std::uint8_t status[]{std::uint8_t(velocity::precision::PromotionStatus::not_requested)};
    velocity::precision::PromotionBuffers buffers{status, 1};
    velocity::precision::promote(p, workspace, initial, source.source(), options, buffers);
    const bool unavailable = status[0] == std::uint8_t(velocity::precision::PromotionStatus::unavailable) &&
        velocity::precision::promotion_count(buffers) == 0 && source.calls == 1;

    auto refined = velocity::reference::second_moments<float>(p, workspace, {0, 1});
    std::array<float, rows> ms{}, mu{};
    for (std::size_t i = 0; i < rows; ++i) {
        ms[i] = refined.spliced[i];
        mu[i] = refined.unspliced[i];
    }
    source.available = true;
    velocity::precision::reapply(p, workspace, refined, source.source(), buffers);
    bool frozen = source.calls == 1;
    for (std::size_t i = 0; i < rows; ++i)
        frozen = frozen && refined.spliced[i] == ms[i] && refined.unspliced[i] == mu[i];
    expect(unavailable && frozen,
           "a source unavailable during initialization stays unpromoted when it becomes available for refinement");
}

void fp16_input_supports_caller_scheduled_float32_blocks() {
    constexpr std::size_t rows = 6, cols = 2;
    std::array<std::uint16_t, rows * cols> hs{}, hu{};
    std::array<float, rows * rows> weights{};
    for (std::size_t i = 0; i < rows; ++i) {
        weights[i * rows + i] = 1;
        for (std::size_t j = 0; j < cols; ++j) {
            const float s = float(i + 1) + .5f * float(j);
            const float u = 1.5f * s + .5f * float(i % 2);
            hs[i * cols + j] = cellerator::numeric::round_f16(s);
            hu[i * cols + j] = cellerator::numeric::round_f16(u);
        }
    }
    auto s = velocity::NumericView::borrow_dense_f16(hs.data(), {rows, cols}, {std::ptrdiff_t(cols), 1});
    auto u = velocity::NumericView::borrow_dense_f16(hu.data(), {rows, cols}, {std::ptrdiff_t(cols), 1});
    auto w = anndatapp::Array::borrow_dense(weights.data(), {rows, rows}, {std::ptrdiff_t(rows), 1});
    velocity::PreparedInput p({s, u, w.view()});
    auto options = fit_options();
    velocity::RunOptions run;
    run.block_genes = 1;
    run.compute_precision = velocity::ComputePrecision::float32;
    velocity::Workspace serial_workspace(p, run);
    auto serial = velocity::reference::steady_state(p, serial_workspace, options);
    auto result = velocity::make_fit_result<float>(cols);
    auto buffers = velocity::fit_buffers<float>(result);
    velocity::Workspace workspace(p, run);
    velocity::validate_run(p);
    std::array<float, cols> score{};
    for (velocity::GeneRange range : {velocity::GeneRange{1, 1}, {0, 1}}) {
        auto m = velocity::reference::moments<float>(p, workspace, range);
        velocity::reference::fit(p, workspace, m, buffers, options);
    }
    result.low_signal = velocity::reference::finalize_selection(
        p, buffers, options, score.data(), score.size());
    auto sg = serial.gamma.view().dense<float>(), rg = result.gamma.view().dense<float>();
    auto so = serial.offset.view().dense<float>(), ro = result.offset.view().dense<float>();
    bool ok = true;
    for (std::size_t j = 0; j < cols; ++j)
        ok = ok && close(sg.data()[j], rg.data()[j]) && close(so.data()[j], ro.data()[j]);
    expect(ok, "caller-scheduled FP32 blocks over mixed FP16/FP32 inputs match the owning reference");
}
}

int main() {
#ifdef VELOCITY_TSAN_TEST
    std::cout << "TSan: allocation interception disabled; checked by the native suite\n";
#endif
    deterministic_blocks_match_owner_and_masks();
    real_threads_match_owner_and_do_not_allocate_after_setup();
    low_signal_recovery_runs_after_every_range();
    stochastic_initialization_and_refit_barriers_match_owner();
    stochastic_low_signal_selection_waits_for_all_ranges();
    unavailable_initial_promotion_stays_frozen_for_refinement();
    fp16_input_supports_caller_scheduled_float32_blocks();
    if (!failures) std::cout << "block tests passed\n";
    return failures ? 1 : 0;
}
