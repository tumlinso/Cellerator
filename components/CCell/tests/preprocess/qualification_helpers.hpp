#pragma once

// Development-only ownership for exercising the allocation-free reference.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qualification {
template<class T> auto dense(const anndatapp::ArrayView& array) {
    auto v = array.dense<T>();
    return cellerator::memory::const_matrix_view<T,std::ptrdiff_t>{v.data(), v.extent(0), v.extent(1),
        v.stride(0), {cellerator::memory::domain::host}, v.stride(1)};
}
template<class T> auto dense(const anndatapp::ArrayEdit& array) {
    auto v = array.dense<T>();
    return cellerator::memory::matrix_view<T,std::ptrdiff_t>{v.data(), v.extent(0), v.extent(1),
        v.stride(0), {cellerator::memory::domain::host}, v.stride(1)};
}

// Byte arrays provide storage in which the kernel may create scalar objects.
struct Scratch {
    std::vector<std::byte> storage;
    void* data;
    explicit Scratch(std::size_t n) : storage(n + alignof(std::max_align_t)-1), data(nullptr) {
        auto p = reinterpret_cast<std::uintptr_t>(storage.data());
        auto aligned = (p + alignof(std::max_align_t)-1) & ~(std::uintptr_t(alignof(std::max_align_t)-1));
        data = storage.data() + (aligned-p);
    }
};
struct Qc {
    std::size_t n, g;
    std::vector<double> cell_total, gene_total, mean, dropout, log_cell_total, log_cell_detected;
    std::vector<double> log_gene_total, log_gene_mean, group_total, group_percent, group_log_total, top_percent, scratch;
    std::vector<std::uint64_t> cell_detected, gene_detected;
    std::vector<std::uint8_t> mask;
    std::size_t tops[3]{2,4,8};
    preprocess::QcOptions options;
    preprocess::QcBuffers buffers;
    template<class View> Qc(View x, std::size_t rows, std::size_t genes, const std::vector<int>& groups) : n(rows), g(genes),
        cell_total(n), gene_total(g), mean(g), dropout(g), log_cell_total(n), log_cell_detected(n),
        log_gene_total(g), log_gene_mean(g), group_total(n), group_percent(n), group_log_total(n), top_percent(n*3), scratch(g),
        cell_detected(n), gene_detected(g), mask(g) {
        for (std::size_t i = 0; i < g; ++i) mask[i] = groups[i] != 0;
        options = {mask.data(), 1, tops, 3};
        buffers.cell_total=cell_total.data(); buffers.cell_detected=cell_detected.data(); buffers.cells_capacity=n;
        buffers.gene_total=gene_total.data(); buffers.gene_detected=gene_detected.data(); buffers.genes_capacity=g;
        buffers.mean=mean.data(); buffers.dropout=dropout.data(); buffers.log_cell_total=log_cell_total.data();
        buffers.log_cell_detected=log_cell_detected.data(); buffers.log_gene_total=log_gene_total.data();
        buffers.log_gene_mean=log_gene_mean.data(); buffers.group_total=group_total.data();
        buffers.group_percent=group_percent.data(); buffers.group_log_total=group_log_total.data(); buffers.group_capacity=n;
        buffers.top_percent=top_percent.data(); buffers.top_capacity=n*3;
        auto s = preprocess::qc(x, options, buffers, scratch.data(), scratch.size());
        if (!s) throw std::runtime_error("QC failed");
    }
};
struct Hvg {
    std::vector<double> mean, dispersion, normalized_dispersion;
    std::vector<std::uint8_t> selected, intersection;
    std::vector<std::size_t> nbatches, batch;
    template<class View> Hvg(View x, std::size_t rows, std::size_t genes,
                             const std::vector<std::int32_t>& batches, preprocess::HvgOptions options) :
      mean(genes), dispersion(genes), normalized_dispersion(genes),
      selected(genes), intersection(genes), nbatches(genes), batch(rows) {
        std::vector<std::int32_t> codes;
        if (options.batched) for (std::size_t i=0; i<batch.size(); ++i) {
            std::size_t j=0; for (; j<codes.size() && codes[j]!=batches[i]; ++j) {}
            if (j==codes.size()) codes.push_back(batches[i]); batch[i]=j;
        }
        std::size_t count=options.batched ? codes.size() : 1, bytes=0;
        auto s=preprocess::hvg_scratch_bytes(genes,count,options,bytes);
        if (s.code!=preprocess::HvgCode::ok) throw std::runtime_error("HVG sizing failed");
        Scratch scratch(bytes);
        preprocess::HvgOutput out{mean.data(),dispersion.data(),normalized_dispersion.data(),selected.data(),intersection.data(),nbatches.data(),genes};
        const auto* b=options.batched ? batch.data() : nullptr;
        if constexpr (requires { x.structure; }) s=preprocess::hvg_sparse(x,b,count,options,out,scratch.data,bytes);
        else s=preprocess::hvg_dense(x,b,count,options,out,scratch.data,bytes);
        if (s.code!=preprocess::HvgCode::ok) throw std::runtime_error("HVG calculation failed");
    }
};
} // namespace qualification
