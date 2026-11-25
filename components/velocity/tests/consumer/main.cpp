#include <Cellerator/memory/view.hh>
#include <stddef.h>
#include <stdint.h>

import cellerator.velocity;

int main() {
    constexpr size_t cells = 4, genes = 1;
    constexpr size_t preparation_capacity = 64;
    constexpr size_t scalar_capacity = 512, mask_capacity = 64;

    float s[cells * genes] = {};
    double u[cells * genes] = {};
    float w[cells * cells] = {};
    const float s_values[cells] = {1, 2, 3, 4};
    const double u_values[cells] = {2, 2, 5, 7};
    for (size_t i = 0; i < cells; ++i) {
        s[i] = s_values[i];
        u[i] = u_values[i];
        w[i * cells + i] = 1;
    }

    using cellerator::memory::const_matrix_view;
    using cellerator::velocity::borrow_matrix;
    cellerator::velocity::Input input{
        borrow_matrix(const_matrix_view<float>{s, cells, genes, genes}),
        borrow_matrix(const_matrix_view<double>{u, cells, genes, genes}),
        borrow_matrix(const_matrix_view<float>{w, cells, cells, cells})};

    cellerator::velocity::PreparationSize needed;
    auto status = cellerator::velocity::preparation_size(input, needed);
    if (!status) return int(status.code);
    size_t indices[preparation_capacity] = {};
    cellerator::velocity::Entry entries[preparation_capacity] = {};
    if (needed.indices > preparation_capacity || needed.entries > preparation_capacity)
        return int(cellerator::velocity::Code::capacity);
    cellerator::velocity::PreparedInput prepared;
    status = cellerator::velocity::prepare(input,
        {indices, preparation_capacity, entries, preparation_capacity}, prepared);
    if (!status) return int(status.code);

    cellerator::velocity::FitOptions fit_options;
    fit_options.trimming = cellerator::velocity::Trimming::none;
    fit_options.min_r2 = -1;
    fit_options.min_ratio = 0;
    cellerator::velocity::RunOptions run_options;
    run_options.block_genes = genes;

#if defined(VELOCITYPP_CONSUMER_STOCHASTIC)
    cellerator::velocity::WorkspaceSize workspace_needed;
    status = cellerator::velocity::workspace_size<float>(prepared, run_options, true, workspace_needed);
    if (!status) return int(status.code);
    float workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(cellerator::velocity::Code::capacity);
    cellerator::velocity::Workspace32 workspace;
    status = cellerator::velocity::bind_workspace<float>(prepared, run_options, true,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    float initial_gamma[genes] = {}, initial_offset[genes] = {}, initial_r2[genes] = {};
    bool selected[genes] = {};
    uint8_t initial_status[genes] = {};
    cellerator::velocity::FitResult32 initialization{
        initial_gamma, initial_offset, initial_r2, selected, initial_status, genes, false};
    float gamma[genes] = {}, offset[genes] = {}, offset2[genes] = {};
    float first_scale[genes] = {}, second_scale[genes] = {}, objective[genes] = {};
    uint8_t fit_status[genes] = {};
    cellerator::velocity::StochasticFitResult32 result{
        initialization, gamma, offset, offset2, first_scale, second_scale,
        objective, fit_status, genes, false};
    double velocity[cells * genes] = {}, second_velocity[cells * genes] = {};
    cellerator::velocity::StochasticOptions options;
    options.initialization = fit_options;
    status = cellerator::velocity::reference::stochastic_into(prepared, workspace, result,
        velocity, cells * genes, second_velocity, cells * genes, genes, options);
    if (!status) return int(status.code);
    if (initial_status[0] != uint8_t(cellerator::velocity::FitStatus::ok) ||
        fit_status[0] != uint8_t(cellerator::velocity::StochasticStatus::ok) ||
        !__builtin_isfinite(gamma[0]) || workspace.rows != cells)
        return 1;
    for (size_t i = 0; i < cells * genes; ++i)
        if (!__builtin_isfinite(velocity[i]) || !__builtin_isfinite(second_velocity[i]))
            return 1;
#else
    cellerator::velocity::WorkspaceSize workspace_needed;
    status = cellerator::velocity::workspace_size<float>(prepared, run_options, false, workspace_needed);
    if (!status) return int(status.code);
    float workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(cellerator::velocity::Code::capacity);
    cellerator::velocity::Workspace32 workspace;
    status = cellerator::velocity::bind_workspace<float>(prepared, run_options, false,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    float gamma[genes] = {}, offset[genes] = {}, r2[genes] = {};
    bool selected[genes] = {};
    uint8_t fit_status[genes] = {};
    cellerator::velocity::FitResult32 result{gamma, offset, r2, selected, fit_status, genes, false};
    double velocity[cells * genes] = {};
    status = cellerator::velocity::reference::steady_state_into(prepared, workspace, result,
        velocity, cells * genes, genes, fit_options);
    if (!status) return int(status.code);
    if (fit_status[0] != uint8_t(cellerator::velocity::FitStatus::ok) || !selected[0] ||
        !__builtin_isfinite(gamma[0]) || workspace.rows != cells)
        return 1;
    for (size_t i = 0; i < cells * genes; ++i)
        if (!__builtin_isfinite(velocity[i])) return 1;
#endif
    return 0;
}
