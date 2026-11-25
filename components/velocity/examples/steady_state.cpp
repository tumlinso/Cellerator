#include <Cellerator/memory/view.hh>
#include <stddef.h>
#include <stdint.h>

import cellerator.velocity;

int main() {
    constexpr size_t cells = 3, genes = 1;
    constexpr size_t preparation_capacity = 64;
    constexpr size_t scalar_capacity = 128, mask_capacity = 64;

    double s[cells * genes] = {};
    double u[cells * genes] = {};
    double w[cells * cells] = {};
    for (size_t i = 0; i < cells; ++i) {
        s[i] = double(i + 1);
        u[i] = 2 * s[i];
        w[i * cells + i] = 1;
    }

    using cellerator::memory::const_matrix_view;
    using cellerator::velocity::borrow_matrix;
    cellerator::velocity::Input input{
        borrow_matrix(const_matrix_view<double>{s, cells, genes, genes}),
        borrow_matrix(const_matrix_view<double>{u, cells, genes, genes}),
        borrow_matrix(const_matrix_view<double>{w, cells, cells, cells})};

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

    cellerator::velocity::RunOptions run_options;
    run_options.block_genes = genes;
    cellerator::velocity::WorkspaceSize workspace_needed;
    status = cellerator::velocity::workspace_size<double>(prepared, run_options, false, workspace_needed);
    if (!status) return int(status.code);
    double workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(cellerator::velocity::Code::capacity);
    cellerator::velocity::Workspace<double> workspace;
    status = cellerator::velocity::bind_workspace<double>(prepared, run_options, false,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    double gamma[genes] = {}, offset[genes] = {}, r2[genes] = {};
    bool selected[genes] = {};
    uint8_t fit_status[genes] = {};
    cellerator::velocity::FitResult result{gamma, offset, r2, selected, fit_status, genes, false};
    double velocity[cells * genes] = {};
    cellerator::velocity::FitOptions fit_options;
    fit_options.trimming = cellerator::velocity::Trimming::none;
    fit_options.min_r2 = -1;
    fit_options.min_ratio = 0;

    status = cellerator::velocity::reference::steady_state_into(prepared, workspace, result,
        velocity, cells * genes, genes, fit_options);
    if (!status) return int(status.code);
    return fit_status[0] == uint8_t(cellerator::velocity::FitStatus::ok) &&
        __builtin_fabs(gamma[0] - 2) < 1e-12 ? 0 : 1;
}
