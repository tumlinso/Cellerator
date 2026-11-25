#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import cellerator.velocity;

#include "support.hh"

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct PreparationStorage {
    std::vector<size_t> indices;
    std::vector<cellerator::velocity::Entry> entries;
    cellerator::velocity::PreparedInput prepared;

    void bind(const cellerator::velocity::Input& input) {
        cellerator::velocity::PreparationSize size{};
        check(bool(cellerator::velocity::preparation_size(input, size)), "preparation sizes native views");
        indices.resize(size.indices);
        entries.resize(size.entries);
        cellerator::velocity::PreparationBuffers buffers{indices.data(), indices.size(), entries.data(), entries.size()};
        check(bool(cellerator::velocity::prepare(input, buffers, prepared)), "prepare binds caller storage");
    }
};

void test_dense_view_and_edit() {
    float source[]{0, 1, 2, 3, 4, 5};
    const double weights[]{1, 0, 0, 1};
    cellerator::memory::matrix_view<float, ptrdiff_t> reversed{source + 3, 2, 2, -3, {}, 2};
    auto s = cellerator::velocity::borrow_matrix(reversed);
    auto u = cellerator::velocity::borrow_matrix(cellerator::memory::as_const(reversed));
    auto w = borrow_dense(weights, {2, 2}, {2, 1});
    PreparationStorage storage;
    storage.bind({s, u, w});

    reversed.data[reversed.column_stride] = 7;
    float ms[4]{}, mu[4]{};
    cellerator::velocity::reference::moments(storage.prepared, 0, 2, ms, mu);
    check(ms[0] == 3 && ms[1] == 7 && ms[2] == 0 && ms[3] == 2 &&
          mu[0] == 3 && mu[1] == 7 && mu[2] == 0 && mu[3] == 2,
          "signed row and sampled column strides preserve logical order and observe caller edits");

    reversed.where.kind = cellerator::memory::domain::device;
    check(cellerator::velocity::validate({cellerator::velocity::borrow_matrix(reversed), u, w}).code == cellerator::velocity::Code::argument,
          "host preparation rejects a view explicitly placed on a device");
}

struct NeighborhoodStorage {
    std::vector<size_t> indices;
    std::vector<cellerator::velocity::Entry> scratch;
    std::vector<float> values;
    std::vector<size_t> columns, offsets;
    size_t used = 0;
};

NeighborhoodStorage prepare_neighborhood(const cellerator::velocity::AcquiredMatrix& input, size_t rows) {
    cellerator::velocity::reference::NeighborhoodSize size{};
    check(bool(cellerator::velocity::reference::neighborhood_size(input, size)),
          "neighborhood sizing accepts the native graph");

    NeighborhoodStorage result;
    result.indices.resize(size.indices);
    result.scratch.resize(size.scratch_entries);
    result.values.resize(size.entries);
    result.columns.resize(size.entries);
    result.offsets.resize(rows + 1);
    cellerator::velocity::reference::NeighborhoodBuffers buffers{
        result.indices.data(), result.indices.size(), result.scratch.data(), result.scratch.size(),
        result.values.data(), result.columns.data(), result.offsets.data(), result.values.size(), result.offsets.size()};
    check(bool(cellerator::velocity::reference::prepare_neighborhood(input, buffers, result.used)),
          "caller-owned buffers prepare the neighborhood graph");
    return result;
}

void check_neighborhood(const NeighborhoodStorage& result) {
    check(result.offsets.size() == 3 && result.used == 3, "normalized graph has three entries in two rows");
    const size_t expected_offsets[]{0, 2, 3}, expected_indices[]{0, 1, 1};
    const float expected_values[]{.5f, .5f, 1.f};
    for (size_t i = 0; i < 3; ++i) {
        check(result.offsets[i] == expected_offsets[i] && result.columns[i] == expected_indices[i],
              "dense, CSR, and CSC inputs produce the expected normalized CSR structure");
        check(std::abs(result.values[i] - expected_values[i]) < 1e-7f,
              "dense, CSR, and CSC inputs produce the expected normalized weights");
    }
}

void test_sparse_view_and_mixed_precision_moments(const NeighborhoodStorage& graph) {
    const float s_values[]{1, 3};
    const int16_t s_indices[]{0, 0};
    const int32_t s_offsets[]{0, 1, 2};
    auto s = borrow_sparse(SparseAxis::row, 2, 1, 2, s_values, s_indices, s_offsets);

    const double u_values[]{10, 30};
    const uint32_t u_indices[]{0, 1};
    const uint64_t u_offsets[]{0, 2};
    auto u = borrow_sparse(SparseAxis::column, 2, 1, 2, u_values, u_indices, u_offsets);

    auto w = borrow_sparse(SparseAxis::row, 2, 2, graph.used, graph.values.data(),
                           graph.columns.data(), graph.offsets.data());
    PreparationStorage storage;
    storage.bind({s, u, w});

    float ms32[2]{}, mu32[2]{};
    double ms64[2]{}, mu64[2]{};
    cellerator::velocity::reference::moments(storage.prepared, 0, 1, ms32, mu32);
    cellerator::velocity::reference::moments(storage.prepared, 0, 1, ms64, mu64);
    check(std::abs(ms32[0] - 2.f) < 1e-6f && std::abs(ms32[1] - 3.f) < 1e-6f &&
          std::abs(mu32[0] - 20.f) < 1e-6f && std::abs(mu32[1] - 30.f) < 1e-6f,
          "FP32 moments consume mixed FP32/FP64 sparse sources and independent integer widths");
    check(std::abs(ms64[0] - 2.) < 1e-12 && std::abs(ms64[1] - 3.) < 1e-12 &&
          std::abs(mu64[0] - 20.) < 1e-12 && std::abs(mu64[1] - 30.) < 1e-12,
          "FP64 moments consume mixed FP32/FP64 sparse sources and independent integer widths");

    auto invalid = borrow_sparse(SparseAxis(255), 2, 1, 2, s_values, s_indices, s_offsets);
    check(cellerator::velocity::validate({invalid, u, w}).code == cellerator::velocity::Code::layout,
          "invalid compressed orientation is rejected");
}
}

int main() {
    test_dense_view_and_edit();
    const float graph[]{0, 2, 0, 0};
    const int32_t csr_indices[]{1}, csc_indices[]{0};
    const int64_t csr_offsets[]{0, 1, 1}, csc_offsets[]{0, 0, 1};
    const auto dense = prepare_neighborhood(borrow_dense(graph, {2, 2}, {2, 1}), 2);
    const auto csr = prepare_neighborhood(borrow_sparse(SparseAxis::row, 2, 2, 1, graph + 1,
                                                       csr_indices, csr_offsets), 2);
    const auto csc = prepare_neighborhood(borrow_sparse(SparseAxis::column, 2, 2, 1, graph + 1,
                                                       csc_indices, csc_offsets), 2);
    check_neighborhood(dense);
    check_neighborhood(csr);
    check_neighborhood(csc);
    test_sparse_view_and_mixed_precision_moments(csr);
}
