#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import cellerator.preprocess;

// Count requested C++ allocation bytes only during the operation. RSS includes
// preparation and allocator retention; these counters do not count direct mmap.
static std::atomic<bool> counting{false};
static std::atomic<std::size_t> calls{0}, bytes{0};
static void count(std::size_t n) {
    if (!counting.load(std::memory_order_relaxed)) return;
    calls.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(n, std::memory_order_relaxed);
}
void* operator new(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    count(n); return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, std::align_val_t a) {
    void* p = nullptr;
    if (posix_memalign(&p, std::size_t(a), n ? n : 1)) throw std::bad_alloc();
    count(n); return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

static long peak_rss() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss; }

template<class T> struct Input {
    std::vector<T> values;
    std::vector<std::int32_t> indices;
    std::vector<std::int64_t> offsets;
};

template<class T>
static Input<T> input(std::size_t n, std::size_t m, bool sparse, std::size_t& nnz) {
    std::minstd_rand rng(1729);
    std::uniform_real_distribution<double> u(0, 1);
    Input<T> x;
    if (sparse) x.offsets.resize(n+1);
    else x.values.resize(n*m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            if (u(rng) > .03 + .17 * double(j % 53) / 52) continue;
            T v = T(1 + (rng() % (3 + j % 37)));
            ++nnz;
            if (sparse) { x.values.push_back(v); x.indices.push_back(std::int32_t(j)); }
            else x.values[i*m + j] = v;
        }
        if (sparse) x.offsets[i+1] = std::int64_t(x.values.size());
    }
    return x;
}

template<class T>
static int run(const std::string& op, const std::string& layout, std::size_t n, std::size_t m) {
    bool sparse = layout == "csr";
    std::size_t nnz = 0;
    auto x = input<T>(n, m, sparse, nnz);
    namespace memory = cellerator::memory;
    namespace tensor = cellerator::tensor::base;
    memory::matrix_view<T> dense{x.values.data(),n,m,m};
    tensor::CompressedView<std::int32_t,T,std::int64_t> compressed{
        {n,m,nnz,tensor::CompressedAxis::row,x.offsets.data(),x.indices.data()},x.values.data()};
    cellerator::preprocess::NormalizationOptions no; no.target_is_set = true; no.target_sum = 1e4;
    std::vector<double> factors(n), row_scratch(n), totals(n), gene_total(m), mean(m), dropout(m), dev(m), qc_scratch(m);
    std::vector<std::uint8_t> included(m); std::vector<std::uint64_t> cell_detected(n), gene_detected(m), stats_scratch(m);
    cellerator::preprocess::NormalizationBuffers norm{factors.data(),n,included.data(),m,row_scratch.data(),n};
    cellerator::preprocess::QcOptions qo;
    cellerator::preprocess::QcBuffers qb; qb.cell_total=totals.data(); qb.cell_detected=cell_detected.data(); qb.cells_capacity=n;
    qb.gene_total=gene_total.data(); qb.gene_detected=gene_detected.data(); qb.genes_capacity=m;
    qb.mean=mean.data(); qb.dropout=dropout.data();
    cellerator::preprocess::ScaleOptions so; so.zero_center=!sparse;
    auto normalize = [&]() { return sparse ? cellerator::preprocess::normalize_into(compressed,compressed,no,norm)
                                           : cellerator::preprocess::normalize_into(memory::as_const(dense),dense,no,norm); };
    auto logarithm = [&]() { return sparse ? cellerator::preprocess::log1p_into(compressed,compressed)
                                          : cellerator::preprocess::log1p_into(memory::as_const(dense),dense); };
    if (op == "hvg" && (!normalize() || !logarithm())) throw std::runtime_error("HVG preprocessing failed");
    cellerator::preprocess::HvgOptions ho; ho.has_top=true; ho.n_top_genes=m/4;
    std::size_t scratch_bytes=0; auto hs=cellerator::preprocess::hvg_scratch_bytes(m,1,ho,scratch_bytes);
    if (hs.code != cellerator::preprocess::HvgCode::ok) throw std::runtime_error("HVG sizing failed");
    std::vector<std::byte> hscratch(scratch_bytes + alignof(std::max_align_t)-1);
    auto address = reinterpret_cast<std::uintptr_t>(hscratch.data());
    auto aligned = (address + alignof(std::max_align_t)-1) & ~(std::uintptr_t(alignof(std::max_align_t)-1));
    void* hworkspace = hscratch.data() + (aligned-address);
    std::vector<double> dispersion(m), normalized_dispersion(m);
    std::vector<std::uint8_t> selected(m), intersection(m); std::vector<std::size_t> batches(m);
    cellerator::preprocess::HvgOutput hout{mean.data(),dispersion.data(),normalized_dispersion.data(),selected.data(),intersection.data(),batches.data(),m};
    long before = peak_rss();
    auto t = std::chrono::steady_clock::now();
    counting.store(true, std::memory_order_relaxed);
    cellerator::preprocess::QcStatus status; std::size_t observations=0;
    if (op == "qc") status = sparse ? cellerator::preprocess::qc(compressed,qo,qb,qc_scratch.data(),qc_scratch.size())
                                     : cellerator::preprocess::qc(memory::as_const(dense),qo,qb,qc_scratch.data(),qc_scratch.size());
    else if (op == "stats") status = sparse ? cellerator::preprocess::feature_stats(compressed,nullptr,mean.data(),dev.data(),m,stats_scratch.data(),m,&observations)
                                            : cellerator::preprocess::feature_stats(memory::as_const(dense),nullptr,mean.data(),dev.data(),m,stats_scratch.data(),m,&observations);
    else if (op == "normalize") status=normalize();
    else if (op == "log1p") status=logarithm();
    else if (op == "scale") status = sparse ? cellerator::preprocess::scale_into(compressed,compressed,so,mean.data(),dev.data(),m,stats_scratch.data(),m)
                                            : cellerator::preprocess::scale_into(memory::as_const(dense),dense,so,mean.data(),dev.data(),m,stats_scratch.data(),m);
    else if (op == "hvg") {
        if (sparse) hs=cellerator::preprocess::hvg_sparse(compressed,nullptr,1,ho,hout,hworkspace,scratch_bytes);
        else hs=cellerator::preprocess::hvg_dense(memory::as_const(dense),nullptr,1,ho,hout,hworkspace,scratch_bytes);
    }
    counting.store(false, std::memory_order_relaxed);
    if (!status || hs.code != cellerator::preprocess::HvgCode::ok) throw std::runtime_error("benchmark operation failed");
    if (op != "baseline" && op != "qc" && op != "stats" && op != "normalize" && op != "log1p" && op != "scale" && op != "hvg") throw std::invalid_argument("unknown operation");
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    auto allocated_calls = calls.load(), allocated_bytes = bytes.load();
    long peak = peak_rss();
    std::size_t input_bytes = sparse ? nnz*(sizeof(T) + sizeof(std::int32_t)) + (n + 1)*sizeof(std::int64_t) : n*m*sizeof(T);
    std::cout << std::setprecision(12) << "{\"operation\":\"" << op << "\",\"layout\":\"" << layout
              << "\",\"dtype\":\"" << (sizeof(T) == 4 ? "float32" : "float64") << "\",\"rows\":" << n
              << ",\"columns\":" << m << ",\"seed\":1729,\"nnz\":" << nnz << ",\"input_bytes\":" << input_bytes
              << ",\"milliseconds\":" << ms << ",\"allocation_calls\":" << allocated_calls
              << ",\"requested_allocation_bytes\":" << allocated_bytes << ",\"baseline_peak_rss_kib\":" << before
              << ",\"peak_rss_kib\":" << peak << ",\"incremental_high_water_kib\":" << peak - before << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("usage: ccell_preprocess_benchmark OP dense|csr float32|float64 ROWS COLS");
        std::string layout = argv[2], dtype = argv[3];
        std::size_t n = std::stoull(argv[4]), m = std::stoull(argv[5]);
        if ((layout != "dense" && layout != "csr") || !n || m < 20 || n > 100000 || m > 100000 || n > 100000000 / m)
            throw std::invalid_argument("invalid layout or bounded geometry");
        if (dtype == "float32") return run<float>(argv[1], layout, n, m);
        if (dtype == "float64") return run<double>(argv[1], layout, n, m);
        throw std::invalid_argument("invalid dtype");
    } catch (const std::exception& e) { counting = false; std::cerr << e.what() << '\n'; return 1; }
}
