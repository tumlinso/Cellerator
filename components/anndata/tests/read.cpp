#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <filesystem>
#include <new>
#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import cellerator.anndata;

namespace ad = anndatapp;
namespace native = cellerator::anndata;
namespace memory = cellerator::memory;
using cellerator::tensor::base::CompressedAxis;

static size_t cpp_allocations;
static bool count_allocations;

void* operator new(size_t bytes) {
    if (count_allocations) ++cpp_allocations;
    if (void* p = malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }

namespace {

constexpr memory::placement host{memory::domain::host};

template<class Operation> size_t allocations(Operation operation) {
    auto before = cpp_allocations;
    count_allocations = true;
    operation();
    count_allocations = false;
    return cpp_allocations - before;
}

template<class Operation>
bool rejects(ad::ErrorCode code, Operation operation) {
    try { operation(); }
    catch (const ad::Error& err) { return err.code() == code; }
    return false;
}

bool dense(const ad::BackedAnnData& file, const char* format) {
    auto selection = file.slice(ad::indices{2, 0, 2}, ad::indices{1, 0}).x();
    auto prepared = selection.prepare_read();
    auto ref = selection.read().view().dense<float>();
    auto owner = native::read_allocated(prepared);
    auto owned = owner.dense<float>();
    if (!rejects(ad::ErrorCode::dtype_mismatch, [&] { (void)owner.dense<double>(); })) return false;

    float supplied[9]{};
    const memory::allocation backing{supplied, sizeof(supplied), alignof(float), host};
    memory::matrix_view<float, ptrdiff_t> out{supplied+6, 3, 2, -3, host};
    const size_t shape[] = {3, 2};
    const ptrdiff_t strides[] = {-3*ptrdiff_t(sizeof(float)), ptrdiff_t(sizeof(float))};
    const ad::DenseBufferView raw{out.data, ad::DType::float32, 2, shape, strides};
    native::read_into(prepared, out, backing);
    prepared.read_into(raw);
    auto raw_allocations = allocations([&] { prepared.read_into(raw); });
    auto native_allocations = allocations([&] { native::read_into(prepared, out, backing); });
    printf("%s dense gather C++ allocations, direct %zu, Cellerator %zu\n", format, raw_allocations, native_allocations);
    if (native_allocations != raw_allocations) return false;
    for (ptrdiff_t i = 0; i < 3; ++i) for (ptrdiff_t j = 0; j < 2; ++j)
        if (out.data[-3*i+j] != ref.data()[2*i+j] || owned.data[2*i+j] != ref.data()[2*i+j]) return false;

    auto short_backing = backing;
    short_backing.bytes = sizeof(float);
    supplied[0] = -42;
    if (!rejects(ad::ErrorCode::invalid_argument, [&] { native::read_into(prepared, out, short_backing); }) ||
        supplied[0] != -42) return false;
    const memory::matrix_view<float, ptrdiff_t> overlapping{supplied, 3, 2, 1, host};
    if (!rejects(ad::ErrorCode::invalid_argument, [&] { native::read_into(prepared, overlapping, backing); }))
        return false;
    const memory::matrix_view<float> packed{supplied, 3, 2, 2, host};
    native::read_into(prepared, packed, backing);
    for (size_t k = 0; k < 6; ++k)
        if (supplied[k] != ref.data()[k]) return false;
    const auto empty = native::read_allocated(file.slice(ad::indices{}).x());
    return empty.rows() == 0 && empty.columns() == 2 && !empty.dense<float>().data;
}

bool sparse(const ad::BackedAnnData& file, const char* format, const char* layer, bool csr) {
    using I = int32_t;
    using O = int64_t;
    auto selection = file.slice(ad::indices{2, 0}, ad::indices{1, 0}).layer(layer);
    auto prepared = selection.prepare_read();
    auto ref_owner = selection.read();
    auto ref = csr ? ref_owner.view().csr<float, I, O>() : ref_owner.view().csc<float, I, O>();
    float values[4]{};
    I indices[4]{};
    O offsets[3]{};
    memory::array_view<float> v{values, 4, host};
    memory::array_view<I> i{indices, 4, host};
    memory::array_view<O> o{offsets, 3, host};
    auto view = native::read_into(prepared, v, i, o);
    const ad::SparseBuffer raw{values, indices, offsets, ad::DType::float32,
                               ad::DType::int32, ad::DType::int64, 4, 4, 3};
    prepared.read_into(raw);
    auto raw_allocations = allocations([&] { prepared.read_into(raw); });
    auto native_allocations = allocations([&] { (void)native::read_into(prepared, v, i, o); });
    printf("%s %s C++ allocations, direct %zu, Cellerator %zu\n", format, layer, raw_allocations, native_allocations);
    if (native_allocations != raw_allocations) return false;
    auto owned = native::read_allocated(selection);
    auto owned_view = owned.compressed<I, float, O>();
    auto axis = csr ? CompressedAxis::row : CompressedAxis::column;
    if (view.structure.axis != axis || owned_view.structure.axis != axis ||
        view.structure.nnz != ref.nnz() || owned_view.structure.nnz != ref.nnz() || view.values != values)
        return false;
    for (size_t k = 0; k < ref.nnz(); ++k)
        if (values[k] != ref.values()[k] || owned_view.values[k] != ref.values()[k] ||
            indices[k] != ref.indices()[k] || owned_view.structure.minor_idx[k] != ref.indices()[k])
            return false;
    for (size_t k = 0; k < 3; ++k)
        if (offsets[k] != ref.offsets()[k] || owned_view.structure.major_ptr[k] != ref.offsets()[k])
            return false;
    o.count = 2;
    values[0] = -42;
    if (!rejects(ad::ErrorCode::invalid_argument, [&] { (void)native::read_into(prepared, v, i, o); }) ||
        values[0] != -42) return false;
    auto empty = native::read_allocated(file.slice(ad::indices{}).layer(layer));
    auto empty_view = empty.compressed<I, float, O>();
    return !empty_view.structure.nnz && empty_view.structure.rows == 0 &&
        empty_view.structure.cols == 2 && empty_view.structure.major_ptr[0] == 0;
}

bool verify(ad::BackedAnnData file, const char* format) {
    if (!dense(file, format) || !sparse(file, format, "csr", true) || !sparse(file, format, "csc", false))
        return false;
    auto half = native::read_allocated(file.layer("half"));
    auto moved = static_cast<native::BufferedMatrix&&>(half);
    file.close();
    auto view = moved.dense<ad::Float16>();
    return view.rows == 3 && view.columns == 2 && view.data[1].bits == 0x8000 && view.data[2].bits == 0x0001;
}

}

int main() {
    char pattern[] = "cellerator-anndata.XXXXXX";
    const char* path = mkdtemp(pattern);
    if (!path) return 1;
    const std::filesystem::path dir{path};
    auto data = ad::AnnData::empty(3, 2);
    const float values[] = {1, 2, 0, 3, 4, 0};
    const float csr_values[] = {1, 2, 3, 4}, csc_values[] = {1, 4, 2, 3};
    const int32_t csr_indices[] = {0, 1, 1, 0}, csc_indices[] = {0, 2, 0, 1};
    const int64_t csr_offsets[] = {0, 2, 3, 4}, csc_offsets[] = {0, 2, 4};
    using Half = ad::Float16;
    const Half half[] = {Half::from_bits(0x3c00), Half::from_bits(0x8000), Half::from_bits(1), {}, {}, {}};
    data.set_x(ad::Array::borrow_dense(values, {3, 2}, {2, 1}));
    data.set_layer("half", ad::Array::borrow_dense(half, {3, 2}, {2, 1}));
    data.set_layer("csr", ad::Array::borrow_sparse(ad::Layout::csr, 3, 2, 4, csr_values, csr_indices, csr_offsets));
    data.set_layer("csc", ad::Array::borrow_sparse(ad::Layout::csc, 3, 2, 4, csc_values, csc_indices, csc_offsets));
    bool ok = true;
#if ANNDATAPP_HAS_HDF5
    auto h5ad = (dir / "sample.h5ad").string();
    ad::write_h5ad(h5ad, data);
    ok = verify(ad::open_h5ad(h5ad), "H5AD") && ok;
#endif
#if ANNDATAPP_HAS_ZARR
    auto zarr = (dir / "sample.zarr").string();
    ad::write_zarr(zarr, data);
    ok = verify(ad::open_zarr(zarr), "Zarr v3") && ok;
    auto zarr_v2 = (dir / "sample-v2.zarr").string();
    ad::ZarrWriteOptions options;
    options.format = ad::ZarrFormat::v2;
    ad::write_zarr(zarr_v2, data, options);
    ok = verify(ad::open_zarr(zarr_v2), "Zarr v2") && ok;
#endif
    std::filesystem::remove_all(dir);
    return ok ? 0 : 2;
}
