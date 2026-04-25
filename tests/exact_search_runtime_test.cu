#include <Cellerator/compute/neighbors/exact_search.hh>

#include "../src/compute/graph/workspace.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

int main() {
    namespace ex = ::cellerator::compute::neighbors::exact_search;
    namespace cg = ::cellerator::compute::graph;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return 1;
    }

    const std::int64_t query_rows = 2;
    const int latent_dim = 2;
    const int top_k = 1;

    const std::vector<__half> query_latent{
        __float2half_rn(1.0f), __float2half_rn(0.0f),
        __float2half_rn(0.0f), __float2half_rn(1.0f)
    };
    const std::vector<float> query_lower{0.11f, 0.16f};
    const std::vector<float> query_upper{10.0f, 10.0f};
    const std::vector<std::int64_t> query_embryo{0, 1};
    const std::vector<std::int64_t> segment_begin{0};
    const std::vector<std::int64_t> segment_end{2};

    cg::device_buffer<__half> d_query_latent(query_latent.size());
    cg::device_buffer<float> d_query_lower(query_lower.size());
    cg::device_buffer<float> d_query_upper(query_upper.size());
    cg::device_buffer<std::int64_t> d_query_embryo(query_embryo.size());
    cg::device_buffer<std::int64_t> d_segment_begin(segment_begin.size());
    cg::device_buffer<std::int64_t> d_segment_end(segment_end.size());
    d_query_latent.upload(query_latent.data(), query_latent.size());
    d_query_lower.upload(query_lower.data(), query_lower.size());
    d_query_upper.upload(query_upper.data(), query_upper.size());
    d_query_embryo.upload(query_embryo.data(), query_embryo.size());
    d_segment_begin.upload(segment_begin.data(), segment_begin.size());
    d_segment_end.upload(segment_end.data(), segment_end.size());

    const ex::ExactSearchQueryDeviceView query_view{
        d_query_latent.data(),
        d_query_lower.data(),
        d_query_upper.data(),
        d_query_embryo.data(),
        query_rows,
        latent_dim
    };

    const std::vector<__half> shard0_latent{
        __float2half_rn(0.90f), __float2half_rn(0.10f),
        __float2half_rn(0.02f), __float2half_rn(0.98f)
    };
    const std::vector<float> shard0_time{0.20f, 0.25f};
    const std::vector<std::int64_t> shard0_embryo{0, 1};
    const std::vector<std::int64_t> shard0_cell{10, 11};

    const std::vector<__half> shard1_latent{
        __float2half_rn(0.98f), __float2half_rn(0.02f),
        __float2half_rn(0.70f), __float2half_rn(0.30f)
    };
    const std::vector<float> shard1_time{0.18f, 0.22f};
    const std::vector<std::int64_t> shard1_embryo{0, 1};
    const std::vector<std::int64_t> shard1_cell{20, 21};

    cg::device_buffer<__half> d_shard0_latent(shard0_latent.size());
    cg::device_buffer<float> d_shard0_time(shard0_time.size());
    cg::device_buffer<std::int64_t> d_shard0_embryo(shard0_embryo.size());
    cg::device_buffer<std::int64_t> d_shard0_cell(shard0_cell.size());
    cg::device_buffer<__half> d_shard1_latent(shard1_latent.size());
    cg::device_buffer<float> d_shard1_time(shard1_time.size());
    cg::device_buffer<std::int64_t> d_shard1_embryo(shard1_embryo.size());
    cg::device_buffer<std::int64_t> d_shard1_cell(shard1_cell.size());
    d_shard0_latent.upload(shard0_latent.data(), shard0_latent.size());
    d_shard0_time.upload(shard0_time.data(), shard0_time.size());
    d_shard0_embryo.upload(shard0_embryo.data(), shard0_embryo.size());
    d_shard0_cell.upload(shard0_cell.data(), shard0_cell.size());
    d_shard1_latent.upload(shard1_latent.data(), shard1_latent.size());
    d_shard1_time.upload(shard1_time.data(), shard1_time.size());
    d_shard1_embryo.upload(shard1_embryo.data(), shard1_embryo.size());
    d_shard1_cell.upload(shard1_cell.data(), shard1_cell.size());

    cg::device_buffer<std::int64_t> d_global_cell(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<std::int64_t> d_global_shard(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<float> d_global_time(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<std::int64_t> d_global_embryo(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<float> d_global_similarity(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<std::int64_t> d_local_cell(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<std::int64_t> d_local_shard(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<float> d_local_time(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<std::int64_t> d_local_embryo(static_cast<std::size_t>(query_rows * top_k));
    cg::device_buffer<float> d_local_similarity(static_cast<std::size_t>(query_rows * top_k));

    const ex::ExactSearchResultDeviceView global_result{
        d_global_cell.data(),
        d_global_shard.data(),
        d_global_time.data(),
        d_global_embryo.data(),
        d_global_similarity.data()
    };
    const ex::ExactSearchResultDeviceView local_result{
        d_local_cell.data(),
        d_local_shard.data(),
        d_local_time.data(),
        d_local_embryo.data(),
        d_local_similarity.data()
    };

    ex::init_result_arrays(global_result, query_rows, top_k);
    ex::init_result_arrays(local_result, query_rows, top_k);
    ex::routed_dense_topk(
        query_view,
        ex::ExactSearchDenseIndexDeviceView{
            d_shard0_latent.data(),
            d_shard0_time.data(),
            d_shard0_embryo.data(),
            d_shard0_cell.data(),
            0
        },
        d_segment_begin.data(),
        d_segment_end.data(),
        1,
        1,
        top_k,
        local_result);
    ex::merge_result_arrays(local_result, global_result, query_rows, top_k);

    ex::init_result_arrays(local_result, query_rows, top_k);
    ex::routed_dense_topk(
        query_view,
        ex::ExactSearchDenseIndexDeviceView{
            d_shard1_latent.data(),
            d_shard1_time.data(),
            d_shard1_embryo.data(),
            d_shard1_cell.data(),
            1
        },
        d_segment_begin.data(),
        d_segment_end.data(),
        1,
        1,
        top_k,
        local_result);
    ex::merge_result_arrays(local_result, global_result, query_rows, top_k);
    cg::cuda_require(cudaDeviceSynchronize(), "exact_search_runtime_test dense sync");

    std::vector<std::int64_t> best_cell(static_cast<std::size_t>(query_rows * top_k));
    std::vector<std::int64_t> best_shard(static_cast<std::size_t>(query_rows * top_k));
    d_global_cell.download(best_cell.data(), best_cell.size());
    d_global_shard.download(best_shard.data(), best_shard.size());
    const bool dense_ok = best_cell[0] == 20 && best_shard[0] == 1 && best_cell[1] == 11 && best_shard[1] == 0;

    const std::vector<__half> sliced_query_latent{
        __float2half_rn(1.0f), __float2half_rn(0.0f), __float2half_rn(1.0f),
        __float2half_rn(0.0f), __float2half_rn(1.0f), __float2half_rn(1.0f)
    };
    const std::vector<float> sliced_lower{0.11f, 0.16f};
    const std::vector<float> sliced_upper{10.0f, 10.0f};
    const std::vector<std::int64_t> sliced_embryo{0, 1};
    cg::device_buffer<__half> d_sliced_query_latent(sliced_query_latent.size());
    cg::device_buffer<float> d_sliced_lower(sliced_lower.size());
    cg::device_buffer<float> d_sliced_upper(sliced_upper.size());
    cg::device_buffer<std::int64_t> d_sliced_embryo(sliced_embryo.size());
    d_sliced_query_latent.upload(sliced_query_latent.data(), sliced_query_latent.size());
    d_sliced_lower.upload(sliced_lower.data(), sliced_lower.size());
    d_sliced_upper.upload(sliced_upper.data(), sliced_upper.size());
    d_sliced_embryo.upload(sliced_embryo.data(), sliced_embryo.size());

    const ex::ExactSearchQueryDeviceView sliced_query_view{
        d_sliced_query_latent.data(),
        d_sliced_lower.data(),
        d_sliced_upper.data(),
        d_sliced_embryo.data(),
        query_rows,
        3
    };

    const std::vector<std::uint32_t> row_slot_offsets{0u, 2u, 4u};
    const std::vector<std::uint32_t> row_widths{2u, 2u};
    const std::vector<std::uint32_t> col_idx{0u, 2u, 1u, 2u};
    const std::vector<__half> values{
        __float2half_rn(0.90f), __float2half_rn(0.90f),
        __float2half_rn(1.00f), __float2half_rn(0.10f)
    };
    const std::vector<float> sliced_time{0.20f, 0.21f};
    const std::vector<std::int64_t> sliced_row_embryo{0, 1};
    const std::vector<std::int64_t> sliced_cell{30, 31};

    cg::device_buffer<std::uint32_t> d_row_slot_offsets(row_slot_offsets.size());
    cg::device_buffer<std::uint32_t> d_row_widths(row_widths.size());
    cg::device_buffer<std::uint32_t> d_col_idx(col_idx.size());
    cg::device_buffer<__half> d_values(values.size());
    cg::device_buffer<float> d_sliced_time(sliced_time.size());
    cg::device_buffer<std::int64_t> d_sliced_row_embryo(sliced_row_embryo.size());
    cg::device_buffer<std::int64_t> d_sliced_cell(sliced_cell.size());
    d_row_slot_offsets.upload(row_slot_offsets.data(), row_slot_offsets.size());
    d_row_widths.upload(row_widths.data(), row_widths.size());
    d_col_idx.upload(col_idx.data(), col_idx.size());
    d_values.upload(values.data(), values.size());
    d_sliced_time.upload(sliced_time.data(), sliced_time.size());
    d_sliced_row_embryo.upload(sliced_row_embryo.data(), sliced_row_embryo.size());
    d_sliced_cell.upload(sliced_cell.data(), sliced_cell.size());

    ex::init_result_arrays(global_result, query_rows, top_k);
    ex::routed_sliced_ell_topk(
        sliced_query_view,
        ex::ExactSearchSlicedEllIndexDeviceView{
            d_row_slot_offsets.data(),
            d_row_widths.data(),
            d_col_idx.data(),
            d_values.data(),
            d_sliced_time.data(),
            d_sliced_row_embryo.data(),
            d_sliced_cell.data(),
            7
        },
        d_segment_begin.data(),
        d_segment_end.data(),
        1,
        1,
        top_k,
        global_result);
    cg::cuda_require(cudaDeviceSynchronize(), "exact_search_runtime_test sliced sync");

    best_cell.assign(best_cell.size(), -1);
    best_shard.assign(best_shard.size(), -1);
    d_global_cell.download(best_cell.data(), best_cell.size());
    d_global_shard.download(best_shard.data(), best_shard.size());
    const bool sliced_ok = best_cell[0] == 30 && best_shard[0] == 7 && best_cell[1] == 31 && best_shard[1] == 7;

    return (dense_ok && sliced_ok) ? 0 : 1;
}
