#include <array>
#include <barrier>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <Cellerator/common/real/fp16.hh>

import velocitypp;
#include "stochastic_support.hh"

namespace {
namespace v = velocitypp;

constexpr size_t rows = 8, source_genes = 3, genes = 5;
constexpr size_t workers = 2;

struct Fixture {
    float spliced[rows * source_genes]{
        1, 3, 2, 2, 4, 3, 1, 2, 4, 3, 7, 5,
        2, 5, 6, 5, 8, 7, 4, 10, 8, 7, 11, 9};
    float unspliced[rows * source_genes]{
        2, 7, 4, 5, 9, 5, 1, 5, 6, 8, 18, 7,
        3, 11, 8, 8, 17, 9, 10, 21, 10, 13, 23, 11};
    std::array<float, rows * rows> weights = [] {
        std::array<float, rows * rows> w{};
        for (size_t i = 0; i < rows; ++i) w[i * rows + i] = 1;
        return w;
    }();
    size_t order[genes]{2, 1, 0, 2, 0};
    uint8_t fit_mask[rows]{1, 1, 1, 0, 1, 1, 1, 1};
    uint8_t gene_mask[source_genes]{1, 1, 0};
    std::array<uint16_t, rows * source_genes> half_unspliced = [this] {
        std::array<uint16_t, rows * source_genes> half;
        for (size_t i = 0; i < half.size(); ++i) {
            half[i] = cellerator::numeric::round_f16(unspliced[i]);
        }
        return half;
    }();
    PreparedStorage prepared{
        borrow_dense(spliced, {rows, source_genes}, {source_genes, 1}),
        borrow_dense(unspliced, {rows, source_genes}, {source_genes, 1}),
        borrow_dense(weights.data(), {rows, rows}, {rows, 1}),
        order, genes, fit_mask, gene_mask};
    PreparedStorage mixed{
        borrow_dense(spliced, {rows, source_genes}, {source_genes, 1}),
        v::borrow_binary16(cellerator::memory::const_matrix_view<uint16_t>{
            half_unspliced.data(), rows, source_genes, source_genes, {}, 1}),
        borrow_dense(weights.data(), {rows, rows}, {rows, 1}),
        order, genes, fit_mask, gene_mask};
};

struct WorkerScratch {
    std::vector<float> scalars;
    std::vector<uint8_t> masks;
    v::Workspace<float> workspace;

    void bind(const v::PreparedInput& p, v::RunOptions options, bool stochastic) {
        v::WorkspaceSize size;
        require_status(v::workspace_size<float>(p, options, stochastic, size));
        scalars.resize(size.scalars);
        masks.resize(size.masks);
        require_status(v::bind_workspace(p, options, stochastic, scalars.data(), scalars.size(),
            masks.data(), masks.size(), workspace));
    }
};

struct FitStorage {
    float gamma[genes]{}, offset[genes]{}, r2[genes]{};
    bool selected[genes]{};
    uint8_t status[genes]{};
    v::BasicFitResult<float> result{gamma, offset, r2, selected, status, genes};
};

struct Output {
    std::array<float, rows * genes> velocity{}, second{};

    static void steady(void* context, const v::BasicMomentBlock<float>& m,
            const v::BasicFitBlock<float>& f) {
        auto& out = *static_cast<Output*>(context);
        v::reference::residuals(m, f, out.velocity.data() + m.first, genes);
    }

    static void stochastic(void* context, const v::BasicStochasticMomentBlock<float>& m,
            const v::BasicStochasticFitBlock<float>& f) {
        auto& out = *static_cast<Output*>(context);
        v::reference::stochastic_residuals(m, f, out.velocity.data() + m.first,
            out.second.data() + m.first, genes);
    }
};

template<class Run> void run_workers(Run run) {
    constexpr v::GeneRange ranges[workers][2]{{{4, 1}, {0, 1}}, {{2, 2}, {1, 1}}};
    std::barrier start{ptrdiff_t(workers)};
    std::array<std::thread, workers> threads;
    std::array<v::Status, workers> statuses{};
    for (size_t i = 0; i < workers; ++i) {
        threads[i] = std::thread([&, i] {
            start.arrive_and_wait();
            for (auto range : ranges[i]) {
                statuses[i] = run(i, range);
                if (!statuses[i]) return;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    for (auto status : statuses) require_status(status);
}

bool same_fit(const v::BasicFitResult<float>& a, const v::BasicFitResult<float>& b) {
    if (a.low_signal != b.low_signal) return false;
    for (size_t i = 0; i < genes; ++i) {
        if (a.gamma[i] != b.gamma[i] || a.offset[i] != b.offset[i] || a.r2[i] != b.r2[i] ||
            a.genes[i] != b.genes[i] || a.status[i] != b.status[i]) return false;
    }
    return true;
}

bool selection_is_global(const v::BasicFitResult<float>& fit) {
    return fit.low_signal && std::abs(fit.gamma[1] - 2) < 1e-6f &&
        std::abs(fit.offset[1] - 1) < 1e-5f &&
        fit.genes[1] && !fit.genes[0] && !fit.genes[2] && !fit.genes[3] && !fit.genes[4] &&
        fit.gamma[2] == fit.gamma[4] && fit.status[2] == fit.status[4];
}

v::FitOptions fit_options() {
    v::FitOptions options;
    options.trimming = v::Trimming::none;
    options.fit_intercept = true;
    options.min_r2 = 1.1;
    options.low_signal = v::LowSignal::scvelo_percentile;
    return options;
}

bool steady(const v::PreparedInput& oracle, const v::PreparedInput& p) {
    v::RunOptions run;
    run.block_genes = 2;
    FitStorage serial, parallel;
    WorkerScratch serial_scratch, coordinator_scratch, worker_scratch[workers];
    serial_scratch.bind(oracle, run, false);
    coordinator_scratch.bind(p, run, false);

    auto options = fit_options();
    Output expected, actual;
    auto& serial_fit = serial.result;
    require_status(v::reference::steady_state_into(oracle, serial_scratch.workspace, serial_fit,
        expected.velocity.data(), expected.velocity.size(), genes, options));

    auto& parallel_fit = parallel.result;
    v::BasicReferenceStages<float> coordinator, stages[workers];
    require_status(v::bind_stages(p, coordinator_scratch.workspace, options, parallel_fit, coordinator));
    for (size_t i = 0; i < workers; ++i) {
        worker_scratch[i].bind(p, run, false);
        require_status(v::bind_worker(coordinator, worker_scratch[i].workspace, stages[i],
            {&actual, Output::steady}));
    }
    if (stages[0].process_block({genes, 1}).code != v::Code::bounds ||
        stages[0].process_block({0, 3}).code != v::Code::capacity) return false;
    run_workers([&](size_t i, v::GeneRange range) {
        return stages[i].process_block(range);
    });
    parallel_fit = coordinator.finish();
    return same_fit(serial_fit, parallel_fit) && selection_is_global(parallel_fit) &&
        expected.velocity == actual.velocity;
}

bool stochastic(const v::PreparedInput& oracle, const v::PreparedInput& p, bool refit_all) {
    v::RunOptions run;
    run.block_genes = 2;
    StochasticStorage<float> serial(oracle, run), parallel(p, run);
    v::StochasticOptions options;
    options.initialization = fit_options();
    if (refit_all) {
        options.initialization.low_signal = v::LowSignal::keep_filters;
        options.refit_all_if_none = true;
    }
    Output expected, actual;
    require_status(v::reference::stochastic_into(oracle, serial.workspace, serial.result,
        expected.velocity.data(), expected.velocity.size(), expected.second.data(),
        expected.second.size(), genes, options));

    WorkerScratch scratch[workers];
    v::BasicStochasticReferenceStages<float> coordinator, stages[workers];
    require_status(v::bind_stages(p, parallel.workspace, options, parallel.result, coordinator));
    for (size_t i = 0; i < workers; ++i) {
        scratch[i].bind(p, run, true);
        require_status(v::bind_worker(coordinator, scratch[i].workspace, stages[i]));
    }
    if (stages[0].process_block({0, 1}).code != v::Code::unfinished) return false;
    run_workers([&](size_t i, v::GeneRange range) {
        return stages[i].initialize_block(range);
    });
    coordinator.resolve_selection();
    for (size_t i = 0; i < workers; ++i) {
        require_status(v::bind_worker(coordinator, scratch[i].workspace, stages[i],
            {&actual, Output::stochastic}));
    }
    if (stages[0].initialize_block({0, 1}).code != v::Code::unfinished) return false;
    run_workers([&](size_t i, v::GeneRange range) {
        return stages[i].process_block(range);
    });
    parallel.result = coordinator.finish();
    auto& initial = parallel.result.initialization;
    if (refit_all) {
        if (parallel.status[2] == uint8_t(v::StochasticStatus::not_selected) ||
            parallel.status[0] != uint8_t(v::StochasticStatus::not_selected) ||
            parallel.status[3] != uint8_t(v::StochasticStatus::not_selected)) return false;
    } else if (!selection_is_global(initial)) return false;

    return same_fit(serial.result.initialization, initial) &&
        serial.result.refit_all == refit_all && parallel.result.refit_all == refit_all &&
        serial.gamma == parallel.gamma && serial.offset == parallel.offset &&
        serial.offset2 == parallel.offset2 && serial.first == parallel.first &&
        serial.second == parallel.second && serial.status == parallel.status &&
        serial.objective == parallel.objective && expected.velocity == actual.velocity &&
        expected.second == actual.second;
}
}

int main() try {
    Fixture fixture;
    const auto& oracle = fixture.prepared.value;
    const auto& mixed = fixture.mixed.value;
    bool steady_ok = steady(oracle, mixed);
    bool stochastic_ok = stochastic(oracle, mixed, false);
    bool refit_ok = stochastic(oracle, mixed, true);
    if (!steady_ok) std::cerr << "steady gene blocks differ from serial or ignore global selection\n";
    if (!stochastic_ok) std::cerr << "stochastic gene blocks differ from serial global selection\n";
    if (!refit_ok) std::cerr << "stochastic gene blocks differ from serial global refit-all\n";
    return !(steady_ok && stochastic_ok && refit_ok);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
