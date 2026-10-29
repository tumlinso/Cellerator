#include <Cellerator/compute/gating/condition_program.hh>
#include <Cellerator/compute/gating/cuda_worksets.cuh>
#include <Cellerator/compute/indexing/cuda_postings.cuh>
#include <Cellerator/compute/indexing/reverse_postings.hh>
#include <Cellerator/compute/support/logical_support.hh>

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace cellerator::compute;

#define CUDA_CHECK(call) do { \
    const auto e = (call); \
    if (e != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); \
} while (0)

struct options {
    std::string label = "e2176_b64_skew09";
    std::uint64_t edges = 2176, cells = 64, entities = 256, seed = 20261027;
    double source_skew = 0.9, gate_density = 0.5;
    bool derivative_heavy = false;
    bool host_only = false;
    std::uint32_t warmup = 3, samples = 21, iterations = 3;
};

std::uint64_t random_next(std::uint64_t &state) {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t x = state;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

double unit_random(std::uint64_t &state) {
    return static_cast<double>(random_next(state) >> 11u)
        / static_cast<double>(std::uint64_t{1} << 53u);
}

std::uint64_t mix_hash(std::uint64_t h, std::uint64_t value) {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return h ^ (value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u));
}

template<class T> struct device_buffer {
    T *data = nullptr;
    std::size_t size = 0;
    device_buffer() = default;
    explicit device_buffer(std::size_t n) : size(n) {
        if (size) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), size * sizeof(T)));
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data) cudaFree(data); }
};

template<class T> void upload(device_buffer<T> &to, const std::vector<T> &from) {
    if (from.size() > to.size) throw std::runtime_error("upload exceeds device buffer");
    if (!from.empty()) CUDA_CHECK(cudaMemcpy(to.data, from.data(), from.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T> std::vector<T> download(const device_buffer<T> &from, std::size_t n) {
    if (n > from.size) throw std::runtime_error("download exceeds device buffer");
    std::vector<T> out(n);
    if (n) CUDA_CHECK(cudaMemcpy(out.data(), from.data, n * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

__device__ __forceinline__ bool item_eligible(
    const indexing::indexed_work_item_v1 &x, std::uint64_t edge_count,
    const std::uint64_t *bitmap) {
    return x.logical_edge_id != 0u && x.logical_edge_ordinal < edge_count
        && ((bitmap[x.logical_edge_ordinal / 64u] >> (x.logical_edge_ordinal % 64u)) & 1u) != 0u;
}

__device__ __forceinline__ std::uint64_t lower_id(
    const indexing::indexed_work_item_v1 *items, std::uint64_t begin,
    std::uint64_t end, std::uint64_t id) {
    std::uint64_t lo = begin, hi = end;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2u;
        if (items[mid].logical_edge_id < id) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

constexpr int kBlockThreads = 256;
using block_scan = cub::BlockScan<unsigned long long, kBlockThreads>;
__global__ void clear_status(gating::cuda_workset_status_v1 *s);

// One CTA owns one cell. It converts each sorted candidate row to exclusive
// eligible prefixes, in bounded tiles so candidate degree can exceed a CTA.
// The derivative prefix contains eligible derivative-only IDs: if the same
// ID is eligible in primal, production merge semantics select the primal item.
__global__ void prefix_worksets_kernel(
    std::uint64_t cells, std::uint64_t edge_count,
    const std::uint64_t *bitmaps, std::uint64_t bitmap_words,
    gating::cuda_candidate_csr_v1 primal,
    gating::cuda_candidate_csr_v1 derivative,
    std::uint64_t *p_prefix, std::uint64_t *d_prefix,
    std::uint64_t *p_counts, std::uint64_t *d_only_counts,
    gating::cuda_workset_status_v1 *status) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    __shared__ block_scan::TempStorage scan_storage;
    __shared__ unsigned long long running;
    __shared__ int row_bad;
    __shared__ int row_valid;
    const auto pb = primal.offsets[cell], pe = primal.offsets[cell + 1u];
    const auto db = derivative.offsets[cell], de = derivative.offsets[cell + 1u];
    const auto *bitmap = bitmap_words == 0u ? nullptr : bitmaps + cell * bitmap_words;

    if (threadIdx.x == 0) {
        row_valid = pb <= pe && pe <= primal.item_count && db <= de
            && de <= derivative.item_count;
        row_bad = !row_valid;
        running = 0u;
        p_counts[cell] = 0u;
        d_only_counts[cell] = 0u;
    }
    __syncthreads();
    if (!row_valid) {
        if (threadIdx.x == 0)
            atomicCAS(&status->code, 0u, static_cast<std::uint32_t>(gating::cuda_workset_code_v1::invalid_argument));
        return;
    }

    for (std::uint64_t tile = pb; tile < pe; tile += kBlockThreads) {
        const auto i = tile + threadIdx.x;
        unsigned long long flag = 0u;
        if (i < pe) {
            const auto x = primal.items[i];
            const bool valid = x.logical_edge_id != 0u && x.logical_edge_ordinal < edge_count
                && (i == pb || primal.items[i - 1u].logical_edge_id < x.logical_edge_id);
            if (!valid) atomicExch(&row_bad, 1);
            flag = valid && bitmap_words != 0u && item_eligible(x, edge_count, bitmap);
        }
        unsigned long long local_prefix = 0u, tile_total = 0u;
        block_scan(scan_storage).ExclusiveSum(flag, local_prefix, tile_total);
        if (i < pe) p_prefix[i] = running + local_prefix;
        __syncthreads();
        if (threadIdx.x == 0) running += tile_total;
        __syncthreads();
    }
    if (threadIdx.x == 0) p_counts[cell] = running;
    __syncthreads();

    if (threadIdx.x == 0) running = 0u;
    __syncthreads();
    for (std::uint64_t tile = db; tile < de; tile += kBlockThreads) {
        const auto i = tile + threadIdx.x;
        unsigned long long flag = 0u;
        if (i < de) {
            const auto x = derivative.items[i];
            const bool valid = x.logical_edge_id != 0u && x.logical_edge_ordinal < edge_count
                && (i == db || derivative.items[i - 1u].logical_edge_id < x.logical_edge_id);
            if (!valid) atomicExch(&row_bad, 1);
            bool duplicate_primal = false;
            if (valid) {
                const auto match = lower_id(primal.items, pb, pe, x.logical_edge_id);
                if (match < pe && primal.items[match].logical_edge_id == x.logical_edge_id)
                    duplicate_primal = item_eligible(primal.items[match], edge_count, bitmap);
            }
            flag = valid && bitmap_words != 0u && item_eligible(x, edge_count, bitmap)
                && !duplicate_primal;
        }
        unsigned long long local_prefix = 0u, tile_total = 0u;
        block_scan(scan_storage).ExclusiveSum(flag, local_prefix, tile_total);
        if (i < de) d_prefix[i] = running + local_prefix;
        __syncthreads();
        if (threadIdx.x == 0) running += tile_total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_only_counts[cell] = running;
        if (row_bad)
            atomicCAS(&status->code, 0u, static_cast<std::uint32_t>(gating::cuda_workset_code_v1::invalid_argument));
    }
}

__global__ void workset_offsets_kernel(
    std::uint64_t cells, const std::uint64_t *p_counts,
    const std::uint64_t *d_only_counts, gating::cuda_output_csr_v1 p_out,
    gating::cuda_output_csr_v1 d_out,
    gating::cuda_workset_status_v1 *status) {
    if (blockIdx.x || threadIdx.x) return;
    if (status->code != 0u) return;
    std::uint64_t p_required = 0u, d_required = 0u;
    for (std::uint64_t c = 0; c < cells; ++c) {
        p_required += p_counts[c];
        d_required += p_counts[c] + d_only_counts[c];
    }
    status->primal_required = p_required;
    status->derivative_required = d_required;
    if (p_required > p_out.item_capacity) {
        status->code = static_cast<std::uint32_t>(gating::cuda_workset_code_v1::primal_overflow);
        return;
    }
    if (d_required > d_out.item_capacity) {
        status->code = static_cast<std::uint32_t>(gating::cuda_workset_code_v1::derivative_overflow);
        return;
    }
    p_out.offsets[0] = d_out.offsets[0] = 0u;
    std::uint64_t ps = 0u, ds = 0u;
    for (std::uint64_t c = 0; c < cells; ++c) {
        ps += p_counts[c];
        ds += p_counts[c] + d_only_counts[c];
        p_out.offsets[c + 1u] = ps;
        d_out.offsets[c + 1u] = ds;
    }
}

__global__ void scatter_parallel_worksets_kernel(
    std::uint64_t cells, std::uint64_t edge_count,
    const std::uint64_t *bitmaps, std::uint64_t bitmap_words,
    gating::cuda_candidate_csr_v1 primal,
    gating::cuda_candidate_csr_v1 derivative,
    const std::uint64_t *p_prefix, const std::uint64_t *d_prefix,
    const std::uint64_t *p_counts, const std::uint64_t *d_only_counts,
    gating::cuda_output_csr_v1 p_out, gating::cuda_output_csr_v1 d_out,
    const gating::cuda_workset_status_v1 *status) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells || status->code != 0u) return;
    const auto pb = primal.offsets[cell], pe = primal.offsets[cell + 1u];
    const auto db = derivative.offsets[cell], de = derivative.offsets[cell + 1u];
    const auto *bitmap = bitmap_words == 0u ? nullptr : bitmaps + cell * bitmap_words;
    const auto p_base = p_out.offsets[cell], d_base = d_out.offsets[cell];
    for (auto k = static_cast<std::uint64_t>(threadIdx.x); k < (pe - pb) + (de - db); k += blockDim.x) {
        if (k < pe - pb) {
            const auto i = pb + k;
            const auto x = primal.items[i];
            if (!item_eligible(x, edge_count, bitmap)) continue;
            const auto p_rank = p_prefix[i];
            p_out.items[p_base + p_rank] = x;
            const auto d_lower = lower_id(derivative.items, db, de, x.logical_edge_id);
            const auto d_before = d_lower == de ? d_only_counts[cell] : d_prefix[d_lower];
            d_out.items[d_base + p_rank + d_before] = x;
        } else {
            const auto j = db + (k - (pe - pb));
            const auto x = derivative.items[j];
            if (!item_eligible(x, edge_count, bitmap)) continue;
            const auto p_lower = lower_id(primal.items, pb, pe, x.logical_edge_id);
            if (p_lower < pe && primal.items[p_lower].logical_edge_id == x.logical_edge_id
                && item_eligible(primal.items[p_lower], edge_count, bitmap)) continue;
            const auto p_before = p_lower == pe ? p_counts[cell] : p_prefix[p_lower];
            d_out.items[d_base + p_before + d_prefix[j]] = x;
        }
    }
}

cudaError_t parallel_build_worksets(
    const gating::condition_program_view_v1 &program, std::uint64_t cells,
    const std::uint64_t *bitmaps, std::uint64_t bitmap_count,
    const gating::cuda_candidate_csr_v1 &p_candidates,
    const gating::cuda_candidate_csr_v1 &d_candidates,
    gating::cuda_output_csr_v1 p_output,
    gating::cuda_output_csr_v1 d_output,
    std::uint64_t *p_prefix, std::uint64_t p_prefix_capacity,
    std::uint64_t *d_prefix, std::uint64_t d_prefix_capacity,
    std::uint64_t *p_counts, std::uint64_t *d_only_counts,
    gating::cuda_workset_status_v1 *status, cudaStream_t stream) {
    const auto words = program.edge_count / 64u + (program.edge_count % 64u != 0u);
    const bool bad_output = cells != 0u && (!p_counts || !d_only_counts
        || !p_output.offsets || !d_output.offsets
        || p_output.offsets_capacity < cells + 1u
        || d_output.offsets_capacity < cells + 1u
        || (p_output.item_capacity && !p_output.items)
        || (d_output.item_capacity && !d_output.items));
    if ((words && cells > std::numeric_limits<std::uint64_t>::max() / words)
        || cells == std::numeric_limits<std::uint64_t>::max()
        || program.structure_epoch == 0u || bitmap_count != cells * words
        || (bitmap_count && bitmaps == nullptr)
        || status == nullptr
        || !p_candidates.offsets || !d_candidates.offsets
        || (p_candidates.item_count && !p_candidates.items)
        || (d_candidates.item_count && !d_candidates.items)
        || p_prefix_capacity < p_candidates.item_count
        || d_prefix_capacity < d_candidates.item_count
        || (p_candidates.item_count && !p_prefix)
        || (d_candidates.item_count && !d_prefix)
        || bad_output) return cudaErrorInvalidValue;
    clear_status<<<1,1,0,stream>>>(status);
    if (cells == 0u) return cudaGetLastError();
    if (cells > 2147483647u) return cudaErrorInvalidValue;
    prefix_worksets_kernel<<<static_cast<unsigned>(cells),kBlockThreads,0,stream>>>(
        cells, program.edge_count, bitmaps, words, p_candidates, d_candidates,
        p_prefix, d_prefix, p_counts, d_only_counts, status);
    auto e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    workset_offsets_kernel<<<1,1,0,stream>>>(cells,p_counts,d_only_counts,
        p_output,d_output,status);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    scatter_parallel_worksets_kernel<<<static_cast<unsigned>(cells),kBlockThreads,0,stream>>>(
        cells, program.edge_count, bitmaps, words, p_candidates, d_candidates,
        p_prefix, d_prefix, p_counts, d_only_counts, p_output, d_output, status);
    return cudaGetLastError();
}

__global__ void clear_status(gating::cuda_workset_status_v1 *s) {
    if (!blockIdx.x && !threadIdx.x) *s = {};
}

struct relation {
    std::vector<support::logical_support_edge_v1> edges;
    std::vector<support::logical_support_participant_v1> participants;
    support::logical_support_view_v1 view{};
};
struct fixture {
    relation base, delta;
    support::support_snapshot_pair_v1 snapshot{};
    std::uint64_t digest = 1469598103934665603ull;
};
std::uint64_t choose_source(std::uint64_t &rng, const options &o) {
    return unit_random(rng) < o.source_skew ? 1u : 1u + random_next(rng) % o.entities;
}
fixture make_fixture(const options &o) {
    fixture f;
    const auto delta_edges = o.edges > 1 ? std::max<std::uint64_t>(1u,o.edges/16u) : 0u;
    auto fill = [&](relation &r,std::uint64_t begin,std::uint64_t count,std::uint64_t relation_id,std::uint64_t &rng) {
        r.edges.reserve(count); r.participants.reserve(count*4u);
        for (std::uint64_t local=0;local<count;++local) {
            const auto ordinal=begin+local;
            const auto source=choose_source(rng,o);
            const auto target=1u+random_next(rng)%o.entities;
            const auto cofactor=1u+random_next(rng)%o.entities;
            const auto participant_begin=r.participants.size();
            r.participants.push_back({0u,0u,source});
            r.participants.push_back({1u,0u,target});
            r.participants.push_back({2u,0u,cofactor});
            const std::uint32_t n=ordinal%31u==0u?4u:3u;
            if(n==4u)r.participants.push_back({0u,0u,source});
            r.edges.push_back({ordinal+1u,participant_begin,n,0u});
            f.digest=mix_hash(f.digest,ordinal+1u); f.digest=mix_hash(f.digest,source);
            f.digest=mix_hash(f.digest,target); f.digest=mix_hash(f.digest,cofactor);
            f.digest=mix_hash(f.digest,n);
        }
        r.view={support::logical_support_schema_version_v1,0u,relation_id,9001u,1u,
            r.edges.data(),r.edges.size(),r.participants.data(),r.participants.size()};
    };
    std::uint64_t rng=o.seed;
    fill(f.base,0u,o.edges-delta_edges,7001u,rng);
    if(delta_edges) fill(f.delta,o.edges-delta_edges,delta_edges,7002u,rng);
    f.snapshot={&f.base.view,delta_edges?&f.delta.view:nullptr,1u};
    if(support::validate_support_snapshot_pair_v1(f.snapshot)!=support::support_snapshot_status_v1::success)
        throw std::runtime_error("support fixture validation failed");
    return f;
}

struct fixture_inputs {
    std::array<gating::condition_node_v1,7> nodes{};
    std::vector<gating::edge_condition_binding_v1> bindings;
    gating::condition_program_view_v1 program{};
    std::vector<std::uint8_t> predicates;
    std::vector<std::uint64_t> bitmap;
    std::vector<indexing::posting_term_v1> p_query,d_query;
    std::uint64_t digest=1469598103934665603ull;
    double gate_density=0;
};
fixture_inputs make_inputs(const options&o) {
    fixture_inputs in;
    in.digest=mix_hash(mix_hash(mix_hash(mix_hash(in.digest,o.edges),o.cells),o.entities),o.seed);
    using gating::condition_opcode_v1;
    in.nodes={{{condition_opcode_v1::predicate,1u,0u,0u,0u,0u,0u},
        {condition_opcode_v1::predicate,1u,0u,0u,0u,1u,0u},
        {condition_opcode_v1::predicate,1u,0u,0u,0u,2u,0u},
        {condition_opcode_v1::predicate,1u,0u,0u,0u,3u,0u},
        {condition_opcode_v1::all_of,1u,0u,0u,2u,0u,0u},
        {condition_opcode_v1::any_of,1u,0u,2u,2u,0u,0u},
        {condition_opcode_v1::all_of,1u,0u,0u,3u,0u,0u}}};
    const auto dirty=std::max<std::uint64_t>(1u,static_cast<std::uint64_t>(std::ceil(o.edges*.01)));
    in.bindings.reserve(o.edges);
    for(std::uint64_t edge=0;edge<o.edges;++edge) {
        std::uint32_t root=0u;
        if(edge<dirty)root=edge%3u==0u?0u:edge%3u==1u?4u:6u;
        else {const std::array<std::uint32_t,4> roots{{1u,2u,3u,5u}};root=roots[(edge-dirty)%roots.size()];}
        in.bindings.push_back({edge,root,0u}); in.digest=mix_hash(in.digest,edge); in.digest=mix_hash(in.digest,root);
    }
    in.program={1u,in.nodes.data(),in.nodes.size(),in.bindings.data(),in.bindings.size(),o.edges,4u,0u};
    in.predicates.resize(o.cells*4u);
    std::uint64_t rng=o.seed^0x94d049bb133111ebull;
    for(auto &v:in.predicates)v=unit_random(rng)<o.gate_density?1u:0u;
    for(auto v:in.predicates)in.digest=mix_hash(in.digest,v);
    in.digest=mix_hash(in.digest,dirty);
    for(std::uint64_t c=0;c<o.cells;++c)for(std::uint64_t e=0;e<dirty;++e)in.digest=mix_hash(in.digest,e);
    rng=o.seed^0xd1b54a32d192ed03ull;
    for(std::uint64_t c=0;c<o.cells;++c) {
        const bool hot=o.source_skew>0.5 && c%4u==0u;
        const auto pq=hot?1u:1u+random_next(rng)%o.entities;
        const auto dq=1u+random_next(rng)%o.entities;
        in.p_query.push_back({0u,0u,pq}); in.d_query.push_back({1u,0u,dq});
    }
    for(const auto &q:in.p_query){in.digest=mix_hash(in.digest,q.role);in.digest=mix_hash(in.digest,q.entity_id);}
    for(const auto &q:in.d_query){in.digest=mix_hash(in.digest,q.role);in.digest=mix_hash(in.digest,q.entity_id);}
    in.bitmap.resize(o.cells*((o.edges+63u)/64u));
std::vector<std::uint8_t> scratch(o.cells*in.nodes.size());
    const auto st=gating::evaluate_condition_bitmaps_v1(in.program,o.cells,
        in.predicates.data(),in.predicates.size(),in.bitmap.data(),in.bitmap.size(),
        scratch.data(),scratch.size());
    if(st!=gating::condition_status_v1::success)throw std::runtime_error("host gate failed");
    for(std::uint64_t c=0;c<o.cells;++c)for(const auto &binding:in.bindings) {
        const auto*p=in.predicates.data()+c*4u; const auto r=binding.root_node;
        in.gate_density+=r==0u?p[0]:r==1u?p[1]:r==2u?p[2]:r==3u?p[3]:r==4u?(p[0]&&p[1]):r==5u?(p[2]||p[3]):(p[0]&&p[1]&&p[2]);
    }
    in.gate_density/=static_cast<double>(o.cells*o.edges);
    return in;
}

struct host_posting_index {
    std::vector<indexing::reverse_posting_record_v1> record_scratch,entries;
    indexing::reverse_posting_view_v1 view{};
};
host_posting_index build_host_index(const fixture&f) {
    const auto req=indexing::query_reverse_posting_requirements_v1(f.snapshot);
    host_posting_index index;index.record_scratch.resize(req.record_capacity);index.entries.resize(req.maximum_unique_entries);
    if(indexing::build_reverse_postings_v1(f.snapshot,index.record_scratch.data(),index.record_scratch.size(),
        index.entries.data(),index.entries.size(),&index.view)!=indexing::posting_status_v1::success)throw std::runtime_error("host posting build failed");
    index.entries.resize(index.view.entry_count);return index;
}
struct host_candidates { std::vector<std::uint64_t> offsets; std::vector<indexing::indexed_work_item_v1> items; };
host_candidates build_candidates(const options&o,const fixture_inputs&in,const host_posting_index&host_index,bool primal) {
    host_candidates out;out.offsets.resize(o.cells+1u);out.offsets[0]=0u;out.items.reserve(o.cells*o.edges);
    std::vector<indexing::indexed_work_item_v1> query_scratch(o.edges*4u);
    std::vector<indexing::posting_term_v1> all_targets;
    if(!primal && o.derivative_heavy)for(std::uint64_t e=1;e<=o.entities;++e)all_targets.push_back({1u,0u,e});
    for(std::uint64_t c=0;c<o.cells;++c) {
        const auto &q=primal?in.p_query[c]:in.d_query[c];
        const auto *terms=(!primal&&o.derivative_heavy)?all_targets.data():&q;
        const auto nterms=(!primal&&o.derivative_heavy)?all_targets.size():1u;
        const auto start=out.items.size();
        indexing::candidate_output_v1 target{out.items.data()+start,o.edges,0u,0u};
        std::vector<indexing::indexed_work_item_v1> row(o.edges);
        target.items=row.data();
        const auto s=indexing::query_reverse_postings_v1(host_index.view,terms,nterms,indexing::posting_match_v1::any,
            {query_scratch.data(),query_scratch.size()},&target);
        if(s!=indexing::posting_status_v1::success)throw std::runtime_error("host posting query failed");
        out.items.insert(out.items.end(),row.begin(),row.begin()+target.written);
        out.offsets[c+1u]=out.items.size();
    }
    return out;
}

struct host_worksets { std::vector<std::uint64_t> po,doff; std::vector<indexing::indexed_work_item_v1> p,d; };
host_worksets oracle(const options&o,const fixture_inputs&in,const host_candidates&p,const host_candidates&d) {
    host_worksets out;out.po.resize(o.cells+1u);out.doff.resize(o.cells+1u);out.po[0]=out.doff[0]=0u;
    const auto words=(o.edges+63u)/64u;
    auto eligible=[&](std::uint64_t cell,const indexing::indexed_work_item_v1&x){return ((in.bitmap[cell*words+x.logical_edge_ordinal/64u]>>(x.logical_edge_ordinal%64u))&1u)!=0u;};
    for(std::uint64_t c=0;c<o.cells;++c) {
        std::map<std::uint64_t,indexing::indexed_work_item_v1> primal_set,derivative_set;
        for(auto k=p.offsets[c];k<p.offsets[c+1u];++k)if(eligible(c,p.items[k])) {
            primal_set.emplace(p.items[k].logical_edge_id,p.items[k]);
            derivative_set.emplace(p.items[k].logical_edge_id,p.items[k]);
        }
        for(auto k=d.offsets[c];k<d.offsets[c+1u];++k)if(eligible(c,d.items[k])) {
            const auto [it,inserted]=derivative_set.emplace(d.items[k].logical_edge_id,d.items[k]);
            if(!inserted&&it->second.logical_edge_ordinal!=d.items[k].logical_edge_ordinal)
                throw std::runtime_error("inconsistent duplicate ordinal");
        }
        for(const auto &item:primal_set)out.p.push_back(item.second);
        for(const auto &item:derivative_set)out.d.push_back(item.second);
        out.po[c+1u]=out.p.size();out.doff[c+1u]=out.d.size();
    }
    return out;
}

bool equal_work(const std::vector<std::uint64_t>&ao,const std::vector<indexing::indexed_work_item_v1>&a,
    const std::vector<std::uint64_t>&bo,const std::vector<indexing::indexed_work_item_v1>&b) {
    if(ao!=bo||a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();++i)if(a[i].logical_edge_id!=b[i].logical_edge_id||a[i].logical_edge_ordinal!=b[i].logical_edge_ordinal)return false;
    return true;
}

struct timing_stats { double median=0,p10=0,p90=0,minimum=0,maximum=0; };
struct event_timer {
    cudaEvent_t a{},b{};
    event_timer(){CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));}
    ~event_timer(){cudaEventDestroy(a);cudaEventDestroy(b);}
    template<class F> double one(std::uint32_t iterations,F f) {
        CUDA_CHECK(cudaEventRecord(a));for(std::uint32_t i=0;i<iterations;++i)f();
        CUDA_CHECK(cudaEventRecord(b));CUDA_CHECK(cudaEventSynchronize(b));float ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));return ms/iterations;
    }
    timing_stats summarize(std::vector<double> v) {
        std::sort(v.begin(),v.end());
        auto pct=[&](double x){const auto pos=x*(v.size()-1u);const auto lo=static_cast<std::size_t>(pos);const auto hi=std::min(lo+1u,v.size()-1u);const auto t=pos-lo;return v[lo]*(1.-t)+v[hi]*t;};
        return {pct(.5),pct(.1),pct(.9),v.front(),v.back()};
    }
    template<class A,class B> std::pair<timing_stats,timing_stats> paired(
        std::uint32_t warmup,std::uint32_t samples,std::uint32_t iterations,A a_fn,B b_fn) {
        for(std::uint32_t i=0;i<warmup;++i){(void)one(iterations,a_fn);(void)one(iterations,b_fn);}
        std::vector<double>a,b;a.reserve(samples);b.reserve(samples);
        for(std::uint32_t i=0;i<samples;++i){
            if(i&1u){b.push_back(one(iterations,b_fn));a.push_back(one(iterations,a_fn));}
            else {a.push_back(one(iterations,a_fn));b.push_back(one(iterations,b_fn));}
        }
        return {summarize(std::move(a)),summarize(std::move(b))};
    }
};

cudaError_t run_parallel(const gating::condition_program_view_v1 &program,
    std::uint64_t cells,const std::uint64_t *bitmaps,std::uint64_t bitmap_count,
    const gating::cuda_candidate_csr_v1 &pc,const gating::cuda_candidate_csr_v1 &dc,
    gating::cuda_output_csr_v1 po,gating::cuda_output_csr_v1 dout,
    std::uint64_t *pp,std::uint64_t ppc,std::uint64_t *dp,std::uint64_t dpc,
    std::uint64_t *pcounts,std::uint64_t *dcounts,
    gating::cuda_workset_status_v1 *status,cudaStream_t stream) {
    return parallel_build_worksets(program,cells,bitmaps,bitmap_count,pc,dc,po,dout,
        pp,ppc,dp,dpc,pcounts,dcounts,status,stream);
}

void run_case(options o) {
    fixture f=make_fixture(o);auto in=make_inputs(o);
    const auto host_index=build_host_index(f);
    const auto p_host=build_candidates(o,in,host_index,true),d_host=build_candidates(o,in,host_index,false);
    const auto expected=oracle(o,in,p_host,d_host);
    const auto all_digest=mix_hash(f.digest,in.digest);

    const auto projection_req=indexing::cuda_posting_projection_requirements_v1(host_index.view);
    std::vector<indexing::posting_term_v1> projection_terms(projection_req.term_count);
    std::vector<std::uint64_t> projection_offsets(projection_req.term_count+1u);
    std::vector<indexing::indexed_work_item_v1> projection_items(projection_req.item_count);
    indexing::cuda_posting_prepare_requirements_v1 projection_actual{};
    if(indexing::build_cuda_posting_projection_v1(host_index.view,projection_terms.data(),projection_terms.size(),
        projection_offsets.data(),projection_offsets.size(),projection_items.data(),projection_items.size(),
        &projection_actual)!=indexing::cuda_posting_prepare_status_v1::success
        ||projection_actual.term_count!=projection_req.term_count||projection_actual.item_count!=projection_req.item_count)
        throw std::runtime_error("host CUDA posting projection preparation failed");

    device_buffer<gating::condition_node_v1> d_nodes(in.nodes.size());upload(d_nodes,std::vector<gating::condition_node_v1>(in.nodes.begin(),in.nodes.end()));
    device_buffer<gating::edge_condition_binding_v1> d_bind(in.bindings.size());upload(d_bind,in.bindings);
    auto program=in.program;program.nodes=d_nodes.data;program.edges=d_bind.data;
    device_buffer<std::uint64_t> d_bitmap(in.bitmap.size());upload(d_bitmap,in.bitmap);

    device_buffer<indexing::posting_term_v1> d_index_terms(projection_terms.size());upload(d_index_terms,projection_terms);
    device_buffer<std::uint64_t> d_index_offsets(projection_offsets.size());upload(d_index_offsets,projection_offsets);
    device_buffer<indexing::indexed_work_item_v1> d_index_items(projection_items.size());upload(d_index_items,projection_items);
    indexing::cuda_reverse_posting_view_v1 device_index{host_index.view.relation_id,host_index.view.structure_id,
        host_index.view.structure_epoch,d_index_terms.data,d_index_offsets.data,projection_terms.size(),
        d_index_items.data,projection_items.size()};

    std::vector<std::uint64_t> hp_query_offsets(o.cells+1u),hd_query_offsets(o.cells+1u);
    std::vector<indexing::posting_term_v1> hd_queries;
    for(std::uint64_t c=0;c<o.cells;++c){
        hp_query_offsets[c]=c;hd_query_offsets[c]=hd_queries.size();
        if(o.derivative_heavy)for(std::uint64_t entity=1;entity<=o.entities;++entity)hd_queries.push_back({1u,0u,entity});
        else hd_queries.push_back(in.d_query[c]);
    }
    hp_query_offsets[o.cells]=o.cells;hd_query_offsets[o.cells]=hd_queries.size();
    device_buffer<std::uint64_t> dp_query_offsets(hp_query_offsets.size()),dd_query_offsets(hd_query_offsets.size());
    device_buffer<indexing::posting_term_v1> dp_queries(in.p_query.size()),dd_queries(hd_queries.size());
    upload(dp_query_offsets,hp_query_offsets);upload(dd_query_offsets,hd_query_offsets);
    upload(dp_queries,in.p_query);upload(dd_queries,hd_queries);
    const auto max_candidate_capacity=o.edges*o.cells;
    if(max_candidate_capacity>std::numeric_limits<std::uint64_t>::max()/2u)throw std::runtime_error("candidate output capacity overflow");
    const auto derivative_capacity=max_candidate_capacity*2u;
    device_buffer<std::uint64_t> d_p_post_offsets(o.cells+1u),d_d_post_offsets(o.cells+1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_post_items(std::max<std::uint64_t>(max_candidate_capacity,1u)),
        d_d_post_items(std::max<std::uint64_t>(max_candidate_capacity,1u));
    device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1> d_p_cursors(in.p_query.size()),d_d_cursors(hd_queries.size());
    device_buffer<std::uint64_t> d_p_query_counts(o.cells),d_d_query_counts(o.cells);
    device_buffer<indexing::cuda_posting_status_v1> d_p_post_status(1u),d_d_post_status(1u);
    indexing::cuda_posting_output_v1 p_post_output{d_p_post_offsets.data,o.cells+1u,d_p_post_items.data,max_candidate_capacity};
    indexing::cuda_posting_output_v1 d_post_output{d_d_post_offsets.data,o.cells+1u,d_d_post_items.data,max_candidate_capacity};
    indexing::cuda_posting_scratch_v1 p_post_scratch{d_p_cursors.data,d_p_cursors.size,d_p_query_counts.data,d_p_post_status.data};
    indexing::cuda_posting_scratch_v1 d_post_scratch{d_d_cursors.data,d_d_cursors.size,d_d_query_counts.data,d_d_post_status.data};
    auto postings_device=[&]{
        CUDA_CHECK(indexing::cuda_query_reverse_postings_any_v1(device_index,program.structure_epoch,o.cells,
            dp_query_offsets.data,dp_queries.data,dp_queries.size,p_post_output,p_post_scratch,nullptr));
        CUDA_CHECK(indexing::cuda_query_reverse_postings_any_v1(device_index,program.structure_epoch,o.cells,
            dd_query_offsets.data,dd_queries.data,dd_queries.size,d_post_output,d_post_scratch,nullptr));
    };

    device_buffer<std::uint64_t> dp_offsets(p_host.offsets.size()),dd_offsets(d_host.offsets.size());
    upload(dp_offsets,p_host.offsets);upload(dd_offsets,d_host.offsets);
    device_buffer<indexing::indexed_work_item_v1> dp_items(std::max<std::size_t>(p_host.items.size(),1u)),dd_items(std::max<std::size_t>(d_host.items.size(),1u));
    upload(dp_items,p_host.items);upload(dd_items,d_host.items);
    gating::cuda_candidate_csr_v1 pc_host{dp_offsets.data,dp_items.data,p_host.items.size()},dc_host{dd_offsets.data,dd_items.data,d_host.items.size()};
    gating::cuda_candidate_csr_v1 pc_device{d_p_post_offsets.data,d_p_post_items.data,max_candidate_capacity},
        dc_device{d_d_post_offsets.data,d_d_post_items.data,max_candidate_capacity};

    const auto p_cap=std::max<std::uint64_t>(max_candidate_capacity,1u),d_cap=std::max<std::uint64_t>(derivative_capacity,1u);
    device_buffer<std::uint64_t> pbo(o.cells+1u),dbo(o.cells+1u),pso(o.cells+1u),dso(o.cells+1u),pcounts(o.cells),dcounts(o.cells);
    device_buffer<indexing::indexed_work_item_v1> p_out(p_cap),d_out(d_cap),p_stage(p_cap),d_stage(d_cap);
    device_buffer<gating::cuda_workset_status_v1> status(1u);
    std::vector<std::uint64_t> h_pso(o.cells+1u),h_dso(o.cells+1u);
    for(std::uint64_t c=0;c<o.cells;++c){
        h_pso[c]=c*o.edges;h_dso[c]=c*2u*o.edges;
    }
    h_pso[o.cells]=max_candidate_capacity;h_dso[o.cells]=derivative_capacity;
    upload(pso,h_pso);upload(dso,h_dso);
    gating::cuda_workset_scratch_v1 scratch{};scratch.primal_offsets=pso.data;scratch.primal_items=p_stage.data;scratch.primal_item_capacity=p_stage.size;
    scratch.derivative_offsets=dso.data;scratch.derivative_items=d_stage.data;scratch.derivative_item_capacity=d_stage.size;
    scratch.primal_counts=pcounts.data;scratch.derivative_counts=dcounts.data;scratch.status=status.data;
    scratch.primal_offsets_capacity=scratch.derivative_offsets_capacity=o.cells+1u;
    gating::cuda_output_csr_v1 po{pbo.data,o.cells+1u,p_out.data,p_cap};
    gating::cuda_output_csr_v1 doutput{dbo.data,o.cells+1u,d_out.data,d_cap};
    device_buffer<std::uint64_t> p_prefix(std::max<std::size_t>(p_host.items.size(),1u)),d_prefix(std::max<std::size_t>(d_host.items.size(),1u));
    device_buffer<std::uint64_t> p_counts(o.cells),d_only_counts(o.cells);
    device_buffer<std::uint8_t> d_predicates(in.predicates.size()),d_node_scratch(o.cells*in.nodes.size());
    upload(d_predicates,in.predicates);
    auto baseline_for=[&](const gating::cuda_candidate_csr_v1&pc,const gating::cuda_candidate_csr_v1&dc){
        CUDA_CHECK(gating::cuda_build_condition_worksets_v1(program,o.cells,d_bitmap.data,in.bitmap.size(),pc,dc,po,doutput,scratch,nullptr));};
    auto candidate_for=[&](const gating::cuda_candidate_csr_v1&pc,const gating::cuda_candidate_csr_v1&dc){
        CUDA_CHECK(run_parallel(program,o.cells,d_bitmap.data,in.bitmap.size(),pc,dc,po,doutput,
            p_prefix.data,max_candidate_capacity,d_prefix.data,max_candidate_capacity,p_counts.data,d_only_counts.data,status.data,nullptr));};
    auto baseline=[&]{baseline_for(pc_host,dc_host);};
    auto candidate=[&]{candidate_for(pc_host,dc_host);};
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    baseline();CUDA_CHECK(cudaDeviceSynchronize());
    auto status_h=download(status,1u)[0];
    auto bp_off=download(pbo,o.cells+1u),bd_off=download(dbo,o.cells+1u);
    auto bp=download(p_out,expected.p.size()),bd=download(d_out,expected.d.size());
    if(status_h.code!=0u||status_h.primal_required!=expected.p.size()||status_h.derivative_required!=expected.d.size()
        ||!equal_work(expected.po,expected.p,bp_off,bp)||!equal_work(expected.doff,expected.d,bd_off,bd))
        throw std::runtime_error("production CUDA baseline disagrees with independent set oracle");
    candidate();CUDA_CHECK(cudaDeviceSynchronize());status_h=download(status,1u)[0];
    auto cp_off=download(pbo,o.cells+1u),cd_off=download(dbo,o.cells+1u);
    auto cp=download(p_out,expected.p.size()),cd=download(d_out,expected.d.size());
    if(status_h.code!=0u||status_h.primal_required!=expected.p.size()||status_h.derivative_required!=expected.d.size()
        ||!equal_work(expected.po,expected.p,cp_off,cp)||!equal_work(expected.doff,expected.d,cd_off,cd))
        throw std::runtime_error("parallel CUDA candidate disagrees with independent set oracle");

    postings_device();CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_p_offsets=download(d_p_post_offsets,o.cells+1u),gpu_d_offsets=download(d_d_post_offsets,o.cells+1u);
    host_candidates gpu_p{gpu_p_offsets,download(d_p_post_items,gpu_p_offsets.back())};
    host_candidates gpu_d{gpu_d_offsets,download(d_d_post_items,gpu_d_offsets.back())};
    if(!equal_work(p_host.offsets,p_host.items,gpu_p.offsets,gpu_p.items)
        ||!equal_work(d_host.offsets,d_host.items,gpu_d.offsets,gpu_d.items))
        throw std::runtime_error("GPU posting retrieval disagrees with exact host candidate rows");
    const auto ps=download(d_p_post_status,1u)[0],ds=download(d_d_post_status,1u)[0];
    if(ps.code||ds.code||ps.required!=p_host.items.size()||ds.required!=d_host.items.size())
        throw std::runtime_error("GPU posting retrieval status/count mismatch");
    baseline_for(pc_device,dc_device);CUDA_CHECK(cudaDeviceSynchronize());status_h=download(status,1u)[0];
    bp_off=download(pbo,o.cells+1u);bd_off=download(dbo,o.cells+1u);
    bp=download(p_out,expected.p.size());bd=download(d_out,expected.d.size());
    if(status_h.code||!equal_work(expected.po,expected.p,bp_off,bp)||!equal_work(expected.doff,expected.d,bd_off,bd))
        throw std::runtime_error("production worksets over GPU postings disagree with set oracle");
    candidate_for(pc_device,dc_device);CUDA_CHECK(cudaDeviceSynchronize());status_h=download(status,1u)[0];
    cp_off=download(pbo,o.cells+1u);cd_off=download(dbo,o.cells+1u);
    cp=download(p_out,expected.p.size());cd=download(d_out,expected.d.size());
    if(status_h.code||!equal_work(expected.po,expected.p,cp_off,cp)||!equal_work(expected.doff,expected.d,cd_off,cd))
        throw std::runtime_error("parallel worksets over GPU postings disagree with set oracle");

    auto check_overflow=[&](bool primal_short){
        const auto required=primal_short?expected.p.size():expected.d.size();
        if(required==0u)throw std::runtime_error("overflow fixture unexpectedly has an empty output");
        const std::uint64_t marker=0xfefefefefefefefeull;
        std::vector<std::uint64_t> sentinel_offsets(o.cells+1u,marker);
        std::vector<indexing::indexed_work_item_v1> sentinel_p(p_out.size,{0xdeadbeefull,0xbeadfeedull});
        std::vector<indexing::indexed_work_item_v1> sentinel_d(d_out.size,{0xdeadbeefull,0xbeadfeedull});
        upload(pbo,sentinel_offsets);upload(dbo,sentinel_offsets);upload(p_out,sentinel_p);upload(d_out,sentinel_d);
        auto short_po=po,short_do=doutput;
        if(primal_short)short_po.item_capacity=required-1u;
        else short_do.item_capacity=required-1u;
        const auto expected_code=primal_short?gating::cuda_workset_code_v1::primal_overflow:gating::cuda_workset_code_v1::derivative_overflow;
        // Exercise both implementations with the same intentionally short view.
        CUDA_CHECK(gating::cuda_build_condition_worksets_v1(program,o.cells,d_bitmap.data,in.bitmap.size(),
            pc_device,dc_device,short_po,short_do,scratch,nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto s=download(status,1u)[0];
        if(s.code!=static_cast<std::uint32_t>(expected_code)||s.primal_required!=expected.p.size()||s.derivative_required!=expected.d.size()
            ||download(pbo,o.cells+1u)!=sentinel_offsets||download(dbo,o.cells+1u)!=sentinel_offsets)
            throw std::runtime_error("production overflow contract failed");
        const auto p_after=download(p_out,p_out.size),d_after=download(d_out,d_out.size);
        for(std::size_t i=0;i<p_after.size();++i)if(p_after[i].logical_edge_id!=sentinel_p[i].logical_edge_id||p_after[i].logical_edge_ordinal!=sentinel_p[i].logical_edge_ordinal)throw std::runtime_error("production overflow partially wrote primal output");
        for(std::size_t i=0;i<d_after.size();++i)if(d_after[i].logical_edge_id!=sentinel_d[i].logical_edge_id||d_after[i].logical_edge_ordinal!=sentinel_d[i].logical_edge_ordinal)throw std::runtime_error("production overflow partially wrote derivative output");
        upload(pbo,sentinel_offsets);upload(dbo,sentinel_offsets);upload(p_out,sentinel_p);upload(d_out,sentinel_d);
        CUDA_CHECK(run_parallel(program,o.cells,d_bitmap.data,in.bitmap.size(),pc_device,dc_device,short_po,short_do,
            p_prefix.data,max_candidate_capacity,d_prefix.data,max_candidate_capacity,p_counts.data,d_only_counts.data,status.data,nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());s=download(status,1u)[0];
        if(s.code!=static_cast<std::uint32_t>(expected_code)||s.primal_required!=expected.p.size()||s.derivative_required!=expected.d.size()
            ||download(pbo,o.cells+1u)!=sentinel_offsets||download(dbo,o.cells+1u)!=sentinel_offsets)
            throw std::runtime_error("parallel overflow contract failed");
        const auto p_candidate_after=download(p_out,p_out.size),d_candidate_after=download(d_out,d_out.size);
        for(std::size_t i=0;i<p_candidate_after.size();++i)if(p_candidate_after[i].logical_edge_id!=sentinel_p[i].logical_edge_id||p_candidate_after[i].logical_edge_ordinal!=sentinel_p[i].logical_edge_ordinal)throw std::runtime_error("parallel overflow partially wrote primal output");
        for(std::size_t i=0;i<d_candidate_after.size();++i)if(d_candidate_after[i].logical_edge_id!=sentinel_d[i].logical_edge_id||d_candidate_after[i].logical_edge_ordinal!=sentinel_d[i].logical_edge_ordinal)throw std::runtime_error("parallel overflow partially wrote derivative output");
        upload(pbo,expected.po);upload(dbo,expected.doff);upload(p_out,expected.p);upload(d_out,expected.d);
    };
    check_overflow(true);check_overflow(false);

    event_timer timer;
    const auto [work_baseline,work_candidate]=timer.paired(o.warmup,o.samples,o.iterations,baseline,candidate);
    const auto [gate_baseline,gate_candidate]=timer.paired(o.warmup,o.samples,o.iterations,[&]{
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(program,o.cells,d_predicates.data,in.predicates.size(),d_bitmap.data,in.bitmap.size(),d_node_scratch.data,d_node_scratch.size,status.data,nullptr));
        baseline();
    },[&]{
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(program,o.cells,d_predicates.data,in.predicates.size(),d_bitmap.data,in.bitmap.size(),d_node_scratch.data,d_node_scratch.size,status.data,nullptr));
        candidate();
    });
    const auto [indexed_baseline,indexed_candidate]=timer.paired(o.warmup,o.samples,o.iterations,[&]{
        postings_device();
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(program,o.cells,d_predicates.data,in.predicates.size(),d_bitmap.data,in.bitmap.size(),d_node_scratch.data,d_node_scratch.size,status.data,nullptr));
        baseline_for(pc_device,dc_device);
    },[&]{
        postings_device();
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(program,o.cells,d_predicates.data,in.predicates.size(),d_bitmap.data,in.bitmap.size(),d_node_scratch.data,d_node_scratch.size,status.data,nullptr));
        candidate_for(pc_device,dc_device);
    });
    auto emit_stats=[](const timing_stats&s){std::cout<<"{\"median\":"<<s.median<<",\"p10\":"<<s.p10<<",\"p90\":"<<s.p90<<",\"min\":"<<s.minimum<<",\"max\":"<<s.maximum<<"}";};
    std::cout<<std::setprecision(9)<<"{\"case\":\""<<o.label<<"\",\"shape\":{\"edges\":"<<o.edges<<",\"cells\":"<<o.cells
      <<",\"source_skew\":"<<o.source_skew<<",\"gate_density\":"<<o.gate_density<<",\"derivative_heavy\":"<<(o.derivative_heavy?"true":"false")
      <<"},\"fixture_digest\":\"0x"<<std::hex<<all_digest<<std::dec<<"\",\"candidate_counts\":{\"p\":"<<p_host.items.size()<<",\"d\":"<<d_host.items.size()
      <<",\"p_eligible\":"<<expected.p.size()<<",\"d_union\":"<<expected.d.size()<<"},\"oracle\":\"exact_production_and_parallel\",\"timings_ms\":{\"workset_production\":";
    emit_stats(work_baseline);std::cout<<",\"workset_parallel\":";emit_stats(work_candidate);
    std::cout<<",\"gate_plus_workset_production\":";emit_stats(gate_baseline);std::cout<<",\"gate_plus_workset_parallel\":";emit_stats(gate_candidate);
    std::cout<<",\"indexed_pipeline_production\":";emit_stats(indexed_baseline);std::cout<<",\"indexed_pipeline_parallel\":";emit_stats(indexed_candidate);
    std::cout<<"},\"samples\":"<<o.samples<<",\"warmup\":"<<o.warmup<<",\"iterations\":"<<o.iterations<<"}\n";
}

options parse(int argc,char**argv) {
    options o;
    auto val=[&](int&i){if(i+1>=argc)throw std::runtime_error("missing option value");return std::string(argv[++i]);};
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--label")o.label=val(i);else if(a=="--edges")o.edges=std::stoull(val(i));else if(a=="--cells")o.cells=std::stoull(val(i));
        else if(a=="--entities")o.entities=std::stoull(val(i));else if(a=="--source-skew")o.source_skew=std::stod(val(i));
        else if(a=="--gate-density")o.gate_density=std::stod(val(i));else if(a=="--seed")o.seed=std::stoull(val(i));
        else if(a=="--derivative-heavy")o.derivative_heavy=true;else if(a=="--host-only")o.host_only=true;else if(a=="--warmup")o.warmup=std::stoul(val(i));
        else if(a=="--samples")o.samples=std::stoul(val(i));else if(a=="--iterations")o.iterations=std::stoul(val(i));
        else throw std::runtime_error("unknown argument: "+a);
    }
    if(!o.edges||!o.cells||o.edges>1000000u||o.cells>4096u||o.entities<3u||o.samples<3u||!o.iterations||o.source_skew<0||o.source_skew>1||o.gate_density<0||o.gate_density>1)
        throw std::runtime_error("invalid probe shape or probabilities");
    return o;
}
}

int main(int argc,char**argv) {
    try {
        const auto o=parse(argc,argv);
        if(o.host_only){
            const auto f=make_fixture(o);const auto in=make_inputs(o);const auto idx=build_host_index(f);
            const auto p=build_candidates(o,in,idx,true),d=build_candidates(o,in,idx,false);const auto w=oracle(o,in,p,d);
            std::cout<<"{\"case\":\""<<o.label<<"\",\"fixture_digest\":\"0x"<<std::hex<<mix_hash(f.digest,in.digest)<<std::dec
                <<"\",\"candidates\":{\"p\":"<<p.items.size()<<",\"d\":"<<d.items.size()<<"},\"eligible\":{\"p\":"<<w.p.size()<<",\"d_union\":"<<w.d.size()<<"},\"host_oracle\":\"pass\"}\n";
        }else run_case(o);
        return 0;
    } catch(const std::exception&e){std::cerr<<"parallel_workset_probe: "<<e.what()<<"\n";return 2;}
}
