#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

import velocity;

using namespace velocity;
using anndatapp::DType;
using anndatapp::Layout;

static std::size_t allocations = 0;
void* operator new(std::size_t n) { ++allocations; if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

static_assert(std::is_trivially_copyable_v<NumericBytes<const std::byte>>);
static_assert(std::is_trivially_copyable_v<NumericBytes<std::byte>>);
static_assert(std::is_trivially_copyable_v<NumericDense<const float>>);
static_assert(std::is_trivially_copyable_v<NumericSparseBytes>);

template<class F> void rejects(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    assert(failed);
}

// The stack buffers remain addressable after the retained owner's destruction.
// Poisoning them makes an early owner release observable even without ASan.
struct GraphBuffers {
    std::uint16_t values[4] = {0, 0x4000, 0, 0};
    std::int32_t indices[1] = {1};
    std::int32_t offsets[3] = {0, 1, 1};
    bool released = false;
};
struct GraphOwner {
    GraphBuffers* graph;
    ~GraphOwner() {
        for (auto& x : graph->values) x = 0;
        for (auto& x : graph->indices) x = 0;
        for (auto& x : graph->offsets) x = 0;
        graph->released = true;
    }
};
NumericView retained_graph(GraphBuffers& graph, Layout layout) {
    auto owner = std::make_shared<GraphOwner>(&graph);
    if (layout == Layout::dense)
        return NumericView::retain_dense_f16(graph.values, {2,2}, {2,1}, std::move(owner));
    graph.values[0] = 0x4000;
    if (layout == Layout::csc) {
        graph.indices[0] = 0;
        graph.offsets[1] = 0;
    }
    SparseF16 buffers{graph.values,1,graph.indices,1,DType::int32,graph.offsets,3,DType::int32};
    return NumericView::retain_sparse_f16(layout,2,2,1,buffers,std::move(owner));
}
std::uint64_t unsigned_at(const std::byte* p, std::size_t width, std::size_t i) {
    p += width*i;
    switch (width) {
    case 1: { std::uint8_t x; std::memcpy(&x,p,1); return x; }
    case 2: { std::uint16_t x; std::memcpy(&x,p,2); return x; }
    case 4: { std::uint32_t x; std::memcpy(&x,p,4); return x; }
    case 8: { std::uint64_t x; std::memcpy(&x,p,8); return x; }
    default: throw std::runtime_error("unexpected sparse integer width");
    }
}
void check(bool ok) { if (!ok) throw std::runtime_error("temporary graph owner regression"); }
void test_temporary_graph_owner(Layout layout) {
    GraphBuffers graph;
    auto result = reference::prepare_neighborhood(retained_graph(graph, layout));
    check(graph.released);
    auto view = result.view();
    check(view.layout() == Layout::csr && view.dtype() == DType::float32);
    auto bytes = view.sparse_bytes();
    check(bytes.rows() == 2 && bytes.columns() == 2 && bytes.nnz() == 3);
    const std::uint64_t expected_offsets[] = {0,2,3}, expected_indices[] = {0,1,1};
    const float expected_values[] = {.5f,.5f,1.f};
    for (std::size_t i = 0; i < 3; ++i) {
        check(unsigned_at(bytes.offsets(),bytes.offset_size(),i) == expected_offsets[i]);
        check(unsigned_at(bytes.indices(),bytes.index_size(),i) == expected_indices[i]);
        float value;
        std::memcpy(&value,bytes.values()+i*sizeof(float),sizeof value);
        check(std::abs(value - expected_values[i]) < 1e-7f);
    }
}

int main() {
    test_temporary_graph_owner(Layout::dense);
    test_temporary_graph_owner(Layout::csr);
    test_temporary_graph_owner(Layout::csc);
    auto owner = std::make_shared<std::vector<std::uint16_t>>(std::initializer_list<std::uint16_t>{0,1,2,3,4,5});
    std::weak_ptr<std::vector<std::uint16_t>> weak = owner;
    auto v = NumericView::retain_dense_f16(owner->data()+3, {2,3}, {-3,1}, owner);
    owner.reset();
    assert(!weak.expired() && v.dtype() == StorageType::float16 && v.rank() == 2);
    auto b = v.bytes();
    assert(b.stride_bytes(0) == -6 && b.stride_bytes(1) == 2 && !b.contiguous());
    auto column = v.subview(1, 2, 1);
    auto col = column.dense<std::uint16_t>();
    assert(col.data()[0] == 5 && col.data()[col.stride(0)] == 2);
    auto reversed = v.subview(0, 1, 2, -1).subview(1, 2, 3, -1);
    auto rev = reversed.dense<std::uint16_t>();
    assert(rev.data()[0] == 2 && rev.stride(0) == 3 && rev.stride(1) == -1);
    rejects([&]{ (void)v.subview(1, 2, 2); });
    rejects([&]{ (void)v.subview(1, 0, 2, -1); });
    rejects([&]{ (void)v.subview(1, 0, 1, 0); });
    rejects([&]{ (void)v.dense<float>(); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{1},{1}); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{1,2},{1}); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{0},{std::numeric_limits<std::ptrdiff_t>::max()}); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{0},{std::numeric_limits<std::ptrdiff_t>::min()}); });
    rejects([&]{ (void)v.subview(0, 0, 1, std::numeric_limits<std::ptrdiff_t>::min()); });
    rejects([&]{ (void)v.subview(0, 0, std::numeric_limits<std::size_t>::max(), -1); });
    auto empty = NumericView::borrow_dense_f16(nullptr,{0,3},{3,1});
    assert(empty.bytes().data() == nullptr && empty.extent(0) == 0);
    assert(empty.subview(1, 1, 2).bytes().data() == nullptr);
    std::uint16_t output[6]{};
    auto edit = NumericEdit::borrow_dense_f16(output,{2,3},{3,1});
    edit.dense<std::uint16_t>().data()[4] = 123;
    assert(edit.view().dense<std::uint16_t>().data()[4] == 123);
    auto array = anndatapp::Array::zeros<float>({2,3});
    NumericEdit e = array.edit();
    e.dense<float>().data()[1] = 7;
    NumericView x = array.view();
    assert(x.dtype() == StorageType::float32 && x.dense<float>().data()[1] == 7);
    auto xx = x.subview(1, 1, 2);
    assert(xx.dense<float>().data()[0] == 7 && xx.extent(1) == 2);
    float f[6] = {}; double d[6] = {};
    auto before = allocations;
    auto fview = NumericView::borrow_dense(f, {2,3}, {3,1});
    auto dview = NumericView::borrow_dense(d, {2,3}, {3,1});
    auto fedit = NumericEdit::borrow_dense(f, {2,3}, {3,1});
    auto dedit = NumericEdit::borrow_dense(d, {2,3}, {3,1});
    auto hview = NumericView::borrow_dense_f16(output, {2,3}, {3,1});
    auto hbytes = hview.bytes(); auto fbytes = fview.bytes(); auto dbytes = dview.bytes();
    auto hsub = hview.subview(1, 1, 2); auto fsub = fview.subview(1, 1, 2); auto dsub = dview.subview(1, 1, 2);
    fedit.dense<float>().data()[0] = 2; dedit.dense<double>().data()[0] = 3;
    assert(allocations == before && hbytes.rank() == 2 && fbytes.data() && dbytes.data());
    assert(hsub.extent(1) == 2 && fsub.extent(1) == 2 && dsub.extent(1) == 2);
    assert(fview.dense<float>().data()[0] == 2 && dview.dense<double>().data()[0] == 3);
    auto refs = v; auto count = weak.use_count(); auto borrowed = refs.bytes();
    assert(weak.use_count() == count && borrowed.data());
    auto ints = anndatapp::Array::zeros<std::uint16_t>({2});
    rejects([&]{ (void)NumericView(ints.view()).dtype(); });

    std::uint16_t values[] = {1,2,3};
    std::int16_t indices[] = {2,0,1};
    std::int32_t offsets[] = {0,2,3};
    SparseF16 s{values,3,indices,3,DType::int16,offsets,3,DType::int32};
    auto csr = NumericView::borrow_sparse_f16(Layout::csr,2,3,3,s);
    csr.validate(); auto sparse = csr.sparse_bytes();
    assert(sparse.nnz() == 3 && sparse.index_size() == 2 && sparse.offset_size() == 4 && sparse.rows() == 2);
    assert(sparse.values_count() == 3 && sparse.indices_count() == 3 && sparse.offsets_count() == 3);
    assert(sparse.index_type() == DType::int16 && sparse.offset_type() == DType::int32);
    auto sparse_allocations = allocations;
    auto borrowed_sparse = NumericView::borrow_sparse_f16(Layout::csr,2,3,3,s);
    auto borrowed_sparse_bytes = borrowed_sparse.sparse_bytes();
    assert(allocations == sparse_allocations && borrowed_sparse_bytes.nnz() == 3);
    rejects([&]{ (void)csr.subview(0, 0, 1); });
    auto bad = s; bad.indices_count = 2;
    rejects([&]{ (void)NumericView::borrow_sparse_f16(Layout::csr,2,3,3,bad); });
    indices[0] = -1; rejects([&]{ csr.validate(); }); indices[0] = 3; rejects([&]{ csr.validate(); }); indices[0] = 2;
    offsets[1] = 4; rejects([&]{ csr.validate(); }); offsets[1] = 2;
    offsets[2] = 2; rejects([&]{ csr.validate(); }); offsets[2] = 3;
    auto held = std::make_shared<int>(1);
    std::weak_ptr<int> sparse_weak = held;
    auto retained = NumericView::retain_sparse_f16(Layout::csr,2,3,3,s,held);
    held.reset();
    auto retained_bytes = retained.sparse_bytes();
    assert(!sparse_weak.expired() && retained_bytes.nnz() == 3);
    rejects([&]{ (void)NumericView::retain_sparse_f16(Layout::csr,2,3,3,s,{}); });
    NumericSparseBytes dangling;
    std::weak_ptr<int> borrow_weak;
    {
        auto own = std::make_shared<int>(1); borrow_weak = own;
        auto scoped = NumericView::retain_sparse_f16(Layout::csr,2,3,3,s,own);
        dangling = scoped.sparse_bytes();
    }
    assert(borrow_weak.expired()); // The descriptor does not retain the owner.
    float fv[] = {1,2,3}; double dv[] = {4,5,6};
    std::uint8_t fi[] = {2,0,1}; std::uint64_t fo[] = {0,2,3};
    SparseBuffers<float> fs{fv,3,fi,3,DType::uint8,fo,3,DType::uint64};
    SparseBuffers<double> ds{dv,3,indices,3,DType::int16,offsets,3,DType::int32};
    auto sparse_before = allocations;
    auto fc = NumericView::borrow_sparse(Layout::csr,2,3,3,fs);
    auto dc = NumericView::borrow_sparse(Layout::csc,3,2,3,ds);
    auto fb = fc.sparse_bytes(); auto db = dc.sparse_bytes();
    assert(allocations == sparse_before);
    assert(fc.dtype() == StorageType::float32 && dc.dtype() == StorageType::float64);
    assert(fb.values() == reinterpret_cast<const std::byte*>(fv) && fb.value_size() == sizeof(float));
    assert(fb.index_type() == DType::uint8 && fb.offset_type() == DType::uint64 && fb.offsets_count() == 3);
    assert(db.values() == reinterpret_cast<const std::byte*>(dv) && db.value_size() == sizeof(double));
    assert(db.index_type() == DType::int16 && db.offset_type() == DType::int32);
    alignas(std::uint64_t) std::byte unaligned_i[sizeof(indices) + 1];
    alignas(std::uint64_t) std::byte unaligned_o[sizeof(offsets) + 1];
    std::memcpy(unaligned_i + 1, indices, sizeof(indices));
    std::memcpy(unaligned_o + 1, offsets, sizeof(offsets));
    auto unaligned = ds; unaligned.indices = unaligned_i + 1; unaligned.offsets = unaligned_o + 1;
    auto unaligned_view = NumericView::borrow_sparse(Layout::csr,2,3,3,unaligned);
    unaligned_view.validate();
    assert(unaligned_view.sparse_bytes().indices() == unaligned_i + 1);
    auto short_fs = fs; short_fs.values_count = 2;
    rejects([&]{ (void)NumericView::borrow_sparse(Layout::csr,2,3,3,short_fs); });
    short_fs = fs; short_fs.offsets_count = 2;
    rejects([&]{ (void)NumericView::borrow_sparse(Layout::csr,2,3,3,short_fs); });
    auto float_owner = std::make_shared<int>(2); std::weak_ptr<int> float_weak = float_owner;
    auto retained_float = NumericView::retain_sparse(Layout::csr,2,3,3,fs,float_owner);
    float_owner.reset(); auto borrowed_float = retained_float.sparse_bytes();
    assert(!float_weak.expired() && borrowed_float.nnz() == 3);
    rejects([&]{ (void)NumericView::retain_sparse(Layout::csc,3,2,3,ds,{}); });
    auto csc = NumericView::borrow_sparse_f16(Layout::csc,3,2,3,s);
    assert(csc.sparse_bytes().columns() == 2);
}
