#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <filesystem>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import cellerator.anndata;
import ccell.preprocess;
import ccell.velocity;

#include "../examples/modeling.hh"

namespace anndata = ::anndatapp;

namespace {

bool half_velocity(cellerator::memory::const_matrix_view<anndata::Float16, ptrdiff_t> half) {
    namespace velocity = ccell::velocity;
    const float weights[] = {1, 0, 0, 1};
    const cellerator::memory::const_matrix_view<float> identity{weights, 2, 2, 2};
    const auto expression = velocity::borrow_binary16(half);
    const velocity::Input input{expression, expression, velocity::borrow_matrix(identity)};
    velocity::PreparedInput prepared;
    velocity::Workspace32 workspace;
    float scratch[32]{}, gamma[2]{}, offset[2]{}, r2[2]{}, residuals[4]{};
    uint8_t mask[2]{}, status[2]{};
    bool selected[2]{};
    velocity::FitResult32 result{gamma, offset, r2, selected, status, 2};
    velocity::FitOptions options;
    options.trimming = velocity::Trimming::none;
    return velocity::prepare(input, {}, prepared) &&
        velocity::bind_workspace(prepared, {}, false, scratch, 32, mask, 2, workspace) &&
        velocity::reference::steady_state_into(prepared, workspace, result, residuals, 4, 2, options) &&
        gamma[0] == 1 && gamma[1] == 1 && residuals[0] == 0 && residuals[1] == 0 &&
        residuals[2] == 0 && residuals[3] == 0;
}

bool model(const anndata::BackedAnnData& data) {
    auto counts = cellerator::anndata::read_allocated(data.x());
    auto half = cellerator::anndata::read_allocated(data.layer("half"));
    const auto features = data.var();
    return features.size() == 2 && features.name(0) == "gene-a" && features.name(1) == "gene-b" &&
        ccell_example::normalized_readout(counts.dense<float>()) &&
        half_velocity(half.dense<anndata::Float16>());
}

}

int main() {
    auto data = anndata::AnnData::empty(2, 2);
    float values[] = {1, 3, 3, 1};
    data.set_x(anndata::Array::borrow_dense(values, {2, 2}, {2, 1}));
    using Half = anndata::Float16;
    const Half half[] = {Half::from_bits(0x3c00), Half::from_bits(0x8000),
        Half::from_bits(0x0001), Half::from_bits(0x4200)};
    data.set_layer("half", anndata::Array::borrow_dense(half, {2, 2}, {2, 1}));
    data.set_var(anndata::Table::with_index({"gene-a", "gene-b"}, "feature_id"));
    char pattern[] = "ccell-modeling.XXXXXX";
    const char* created = mkdtemp(pattern);
    if (!created) return 2;
    const std::filesystem::path directory = created;
#if ANNDATAPP_HAS_HDF5
    const auto h5ad = (directory / "counts.h5ad").string();
    anndata::write_h5ad(h5ad, data);
    if (!model(anndata::open_h5ad(h5ad))) return 3;
#endif
#if ANNDATAPP_HAS_ZARR
    const auto zarr = (directory / "counts.zarr").string();
    anndata::write_zarr(zarr, data);
    if (!model(anndata::open_zarr(zarr))) return 4;
#endif
    std::filesystem::remove_all(directory);
}
