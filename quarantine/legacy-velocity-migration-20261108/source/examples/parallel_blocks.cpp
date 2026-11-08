#include <array>
#include <cstdio>
#include <exception>
#include <thread>

import velocity;

int main() {
    constexpr std::size_t rows = 6, genes = 3;
    auto s = anndatapp::Array::empty<double>({rows, genes});
    auto u = anndatapp::Array::empty<double>({rows, genes});
    auto w = anndatapp::Array::zeros<double>({rows, rows});
    auto* spliced = s.edit().dense<double>().data();
    auto* unspliced = u.edit().dense<double>().data();
    auto* weights = w.edit().dense<double>().data();
    for (std::size_t i = 0; i < rows; ++i) {
        weights[i * rows + i] = 1;
        for (std::size_t j = 0; j < genes; ++j) {
            spliced[i * genes + j] = 1 + double(i) + .25 * double(j);
            unspliced[i * genes + j] = 1.4 * spliced[i * genes + j] + .1 * double(i % 2);
        }
    }

    velocity::PreparedInput input({s.view(), u.view(), w.view()});
    velocity::FitOptions options;
    options.trimming = velocity::Trimming::none;
    options.fit_intercept = true;
    options.min_r2 = 0;
    options.min_ratio = 0;

    auto result = velocity::make_fit_result<double>(genes);
    auto buffers = velocity::fit_buffers<double>(result);
    auto output_array = anndatapp::Array::zeros<double>({rows, genes});
    auto output = velocity::acquire_output(input, output_array.edit());
    velocity::RunOptions run;
    run.block_genes = 2;
    velocity::validate_run(input);

    constexpr std::array<velocity::GeneRange, 2> schedule{{{2, 1}, {0, 2}}};
    std::array<velocity::Workspace, schedule.size()> workspaces{
        velocity::Workspace(input, run), velocity::Workspace(input, run)};
    std::array<std::exception_ptr, schedule.size()> errors{};
    std::array<std::thread, schedule.size()> workers;
    for (std::size_t k = 0; k < schedule.size(); ++k)
        workers[k] = std::thread([&, k] {
            try {
                auto moments = velocity::reference::moments<double>(input, workspaces[k], schedule[k]);
                auto fit = velocity::reference::fit(input, workspaces[k], moments, buffers, options);
                velocity::reference::write_residuals(moments, fit, output);
            } catch (...) { errors[k] = std::current_exception(); }
        });
    for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) return 1;
    std::array<double, genes> score{};
    result.low_signal = velocity::reference::finalize_selection(
        input, buffers, options, score.data(), score.size());

    auto gamma = result.gamma.view().dense<double>();
    auto selected = result.genes.view().dense<bool>();
    for (std::size_t j = 0; j < genes; ++j)
        std::printf("gene %zu: gamma %.5f, selected %s\n", j, gamma.data()[j],
                    selected.data()[j] ? "yes" : "no");
    return 0;
}
