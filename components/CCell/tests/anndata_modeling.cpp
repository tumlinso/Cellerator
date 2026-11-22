#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <filesystem>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import anndata;
import ccell.interop.anndata;
import ccell.preprocess;

#include "../examples/modeling.hh"

namespace anndata = ::anndatapp;

namespace {

bool model(const anndata::AnnData& data) {
    const auto x = data.x();
    const auto features = data.var();
    cellerator::memory::const_matrix_view<float, ptrdiff_t> counts;
    return features.size() == 2 && features.name(0) == "gene-a" && features.name(1) == "gene-b" &&
        ccell::io::borrow_dense(x, counts) == ccell::io::BorrowCode::ok &&
        counts.data == x.dense<float>().data() && ccell_example::normalized_readout(counts);
}

bool bindings(const anndata::AnnData& data) {
    using namespace cellerator::tensor::base;
    using ccell::io::BorrowCode;
    using anndata::Axis;
    const auto x = data.x();
    const auto reversed = x.select({Axis::stride(1, 2, 1, true), Axis::all()});
    const auto gathered = x.select({Axis::gather({1, 0, 1}), Axis::all()});
    cellerator::memory::const_matrix_view<float, ptrdiff_t> dense;
    if (ccell::io::borrow_dense(reversed, dense) != BorrowCode::ok ||
        dense.row_stride != -2 || dense.data != x.dense<float>().data()+2) return false;
    if (ccell::io::borrow_dense(gathered, dense) != BorrowCode::non_contiguous || dense.data) return false;
    cellerator::memory::const_matrix_view<double, ptrdiff_t> wrong_type;
    if (ccell::io::borrow_dense(x, wrong_type) != BorrowCode::dtype) return false;

    float values[] = {2, 4};
    int32_t indices[] = {0, 1};
    int64_t offsets[] = {0, 1, 2};
    auto csr = anndata::Array::borrow_sparse(anndata::Layout::csr, 2, 2, 2, values, indices, offsets);
    CompressedView<int32_t, const float, int64_t> sparse;
    if (ccell::io::borrow_compressed(csr.view(), CompressedAxis::row, sparse) != BorrowCode::ok ||
        sparse.structure.major_ptr != offsets || sparse.structure.minor_idx != indices ||
        sparse.values != values) return false;
    if (ccell::io::borrow_compressed(csr.view(), CompressedAxis::column, sparse) != BorrowCode::layout ||
        sparse.values) return false;
    CompressedView<int32_t, float, int64_t> editable;
    if (ccell::io::borrow_compressed(csr.edit(), CompressedAxis::row, editable) != BorrowCode::ok) return false;
    editable.values[1] = 7;
    if (values[1] != 7) return false;

    const anndata::ArrayEdit unbound;
    cellerator::memory::matrix_view<float, ptrdiff_t> editable_dense{values, 1, 2, 2};
    return ccell::io::borrow_dense(unbound, editable_dense) == BorrowCode::invalid &&
        !editable_dense.data &&
        ccell::io::borrow_compressed(unbound, CompressedAxis::row, editable) == BorrowCode::invalid &&
        !editable.values;
}

}

int main() {
    auto data = anndata::AnnData::empty(2, 2);
    float values[] = {1, 3, 3, 1};
    data.set_x(anndata::Array::borrow_dense(values, {2, 2}, {2, 1}));
    data.set_var(anndata::Table::with_index({"gene-a", "gene-b"}, "feature_id"));
    if (!model(data) || !bindings(data)) return 1;
    char pattern[] = "ccell-modeling.XXXXXX";
    const char* created = mkdtemp(pattern);
    if (!created) return 2;
    const std::filesystem::path directory = created;
#if ANNDATAPP_HAS_HDF5
    const auto h5ad = (directory / "counts.h5ad").string();
    anndata::write_h5ad(h5ad, data);
    if (!model(anndata::read_h5ad(h5ad))) return 3;
#endif
#if ANNDATAPP_HAS_ZARR
    const auto zarr = (directory / "counts.zarr").string();
    anndata::write_zarr(zarr, data);
    if (!model(anndata::read_zarr(zarr))) return 4;
#endif
    std::filesystem::remove_all(directory);
}
