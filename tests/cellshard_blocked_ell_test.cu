#include "../extern/CellShard/include/CellShard/CellShard.hh"
#include "../extern/CellShard/src/convert/blocked_ell_from_compressed.cuh"
#include "../extern/CellShard/src/convert/sliced_ell_from_blocked_ell.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace cs = ::cellshard;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void fill_compressed_part(cs::sparse::compressed *part,
                          std::uint32_t rows,
                          std::uint32_t cols,
                          const std::uint32_t *major_ptr,
                          const std::uint32_t *minor_idx,
                          const float *values) {
    cs::sparse::init(part, rows, cols, major_ptr[rows], cs::sparse::compressed_by_row);
    require(cs::sparse::allocate(part) != 0, "compressed allocate failed");
    for (std::uint32_t i = 0; i <= rows; ++i) part->majorPtr[i] = major_ptr[i];
    for (std::uint32_t i = 0; i < part->nnz; ++i) {
        part->minorIdx[i] = minor_idx[i];
        part->val[i] = __float2half(values[i]);
    }
}

bool close_half(__half lhs, float rhs, float tol = 1.0e-3f) {
    return std::fabs(__half2float(lhs) - rhs) <= tol;
}

} // namespace

int main() {
    const std::uint32_t major0[] = { 0u, 2u, 3u };
    const std::uint32_t minor0[] = { 0u, 3u, 1u };
    const float values0[] = { 1.0f, 2.0f, 3.0f };

    const std::uint32_t major1[] = { 0u, 2u, 3u };
    const std::uint32_t minor1[] = { 0u, 2u, 3u };
    const float values1[] = { 4.0f, 5.0f, 6.0f };

    cs::sparse::compressed *part0 = new cs::sparse::compressed();
    cs::sparse::compressed *part1 = new cs::sparse::compressed();
    fill_compressed_part(part0, 2u, 4u, major0, minor0, values0);
    fill_compressed_part(part1, 2u, 4u, major1, minor1, values1);

    cs::sharded<cs::sparse::compressed> src;
    cs::init(&src);
    require(cs::append_partition(&src, part0) != 0, "append part0 failed");
    require(cs::append_partition(&src, part1) != 0, "append part1 failed");

    const unsigned int candidates[] = { 2u, 4u };
    cs::convert::blocked_ell_tune_result tune = {};
    require(cs::convert::choose_blocked_ell_block_size(src.parts[0], candidates, 2u, &tune) != 0, "choose_blocked_ell_block_size failed");
    require(tune.block_size == 2u || tune.block_size == 4u, "unexpected block size");
    {
        const std::uint32_t uniform_rows[64] = {
            4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
            4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
            4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
            4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u
        };
        const std::uint32_t skew_rows[64] = {
            32u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
            1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
            1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
            1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u
        };
        const unsigned int sliced_candidates[] = { 8u, 16u, 32u, 64u };
        cs::convert::sliced_ell_tune_result sliced_tune{};
        require(cs::convert::choose_sliced_ell_slice_rows(uniform_rows, 64u, 256u, sliced_candidates, 4u, &sliced_tune) != 0,
                "choose_sliced_ell_slice_rows uniform failed");
        require(sliced_tune.slice_rows == 64u, "uniform sliced ell tuner should prefer one large slice");
        require(cs::convert::choose_sliced_ell_slice_rows(skew_rows, 64u, 95u, sliced_candidates, 4u, &sliced_tune) != 0,
                "choose_sliced_ell_slice_rows skew failed");
        require(sliced_tune.slice_rows == 8u, "skewed sliced ell tuner should prefer smaller slices");
    }

    cs::sparse::blocked_ell local;
    cs::sparse::init(&local);
    require(cs::convert::blocked_ell_from_compressed(src.parts[0], 2u, &local) != 0, "blocked_ell_from_compressed failed");
    require(local.block_size == 2u, "blocked ell block size mismatch");
    require(local.ell_cols == 4u, "blocked ell ell_cols mismatch");
    require(local.blockColIdx[0] == 0u && local.blockColIdx[1] == 1u, "row block 0 columns mismatch");
    require(close_half(local.val[0], 1.0f), "blocked ell row0 slot0 col0 mismatch");
    require(close_half(local.val[3], 2.0f), "blocked ell row0 slot1 col1 mismatch");
    require(close_half(local.val[5], 3.0f), "blocked ell row1 slot0 col1 mismatch");
    {
        const unsigned int sliced_candidates[] = { 1u, 2u, 4u };
        cs::convert::sliced_ell_tune_result sliced_tune{};
        cs::sparse::sliced_ell sliced;
        cs::sparse::init(&sliced);
        require(cs::convert::sliced_ell_from_blocked_ell_auto(&local, sliced_candidates, 3u, &sliced, &sliced_tune) != 0,
                "sliced_ell_from_blocked_ell_auto failed");
        require(sliced_tune.slice_rows == 1u, "small blocked ell should pick row-local slices");
        require(cs::sparse::uniform_slice_rows(&sliced) == 1u, "uniform sliced ell rows mismatch");
        require(sliced.slice_count == 2u, "sliced ell slice count mismatch");
        require(sliced.slice_widths[0] == 2u && sliced.slice_widths[1] == 1u, "sliced ell widths mismatch");
        require(sliced.col_idx[0] == 0u && sliced.col_idx[1] == 3u, "sliced ell first row columns mismatch");
        require(close_half(sliced.val[0], 1.0f) && close_half(sliced.val[1], 2.0f), "sliced ell first row values mismatch");
        require(sliced.col_idx[2] == 1u && close_half(sliced.val[2], 3.0f), "sliced ell second row mismatch");
        cs::sparse::clear(&sliced);
    }

    cs::sharded<cs::sparse::blocked_ell> blocked;
    cs::init(&blocked);
    require(cs::convert::repack_sharded_compressed_to_blocked_ell(&src, 2u, 1ul, &blocked) != 0, "repack_sharded_compressed_to_blocked_ell failed");
    require(blocked.num_partitions == 2ul, "blocked ell repack part count mismatch");
    require(blocked.partition_aux[0] == cs::sparse::pack_blocked_ell_aux(2u, 2ul), "blocked ell aux mismatch");
    require(cs::device::set_shards_by_device_bytes(&blocked, 128u) != 0, "set_shards_by_device_bytes failed");
    require(blocked.num_shards >= 1ul, "blocked ell shard count mismatch");

    cs::device::partition_record<cs::sparse::blocked_ell> record;
    cs::device::zero_record(&record);
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");
    require(cs::device::upload(blocked.parts[0], &record) == cudaSuccess, "blocked ell upload failed");
    require(record.view != 0 && record.a0 != 0 && record.a1 != 0, "blocked ell upload pointers missing");
    require(cs::device::release(&record) == cudaSuccess, "blocked ell release failed");

    {
        char path[] = "/tmp/cellshard_blocked_ell_cacheXXXXXX";
        const int fd = ::mkstemp(path);
        require(fd >= 0, "mkstemp failed");
        ::close(fd);
        std::remove(path);
        const std::string out_path = std::string(path) + ".csh5";
        const std::string cache_root = std::string(path) + ".cache";
        std::vector<std::uint64_t> partition_rows(blocked.num_partitions, 0u);
        std::vector<std::uint64_t> partition_nnz(blocked.num_partitions, 0u);
        std::vector<std::uint64_t> partition_aux(blocked.num_partitions, 0u);
        std::vector<std::uint64_t> partition_row_offsets(blocked.num_partitions + 1u, 0u);
        std::vector<std::uint32_t> partition_dataset_ids(blocked.num_partitions, 0u);
        std::vector<std::uint32_t> partition_codec_ids(blocked.num_partitions, 0u);
        std::vector<std::uint64_t> shard_offsets(blocked.num_shards + 1u, 0u);
        cs::dataset_codec_descriptor codec{};
        cs::dataset_layout_view layout{};
        for (unsigned long i = 0; i < blocked.num_partitions; ++i) {
            partition_rows[i] = (std::uint64_t) blocked.partition_rows[i];
            partition_nnz[i] = (std::uint64_t) blocked.partition_nnz[i];
            partition_aux[i] = (std::uint64_t) blocked.partition_aux[i];
            partition_row_offsets[i] = (std::uint64_t) blocked.partition_offsets[i];
        }
        partition_row_offsets[blocked.num_partitions] = (std::uint64_t) blocked.rows;
        for (unsigned long i = 0; i <= blocked.num_shards; ++i) {
            shard_offsets[i] = (std::uint64_t) blocked.shard_offsets[i];
        }

        codec.codec_id = 0u;
        codec.family = cs::dataset_codec_family_sliced_ell;
        codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
        codec.scale_value_code = 0u;
        codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
        codec.flags = 0u;

        layout.rows = (std::uint64_t) blocked.rows;
        layout.cols = (std::uint64_t) blocked.cols;
        layout.nnz = (std::uint64_t) blocked.nnz;
        layout.num_partitions = (std::uint64_t) blocked.num_partitions;
        layout.num_shards = (std::uint64_t) blocked.num_shards;
        layout.partition_rows = partition_rows.data();
        layout.partition_nnz = partition_nnz.data();
        layout.partition_axes = nullptr;
        layout.partition_aux = partition_aux.data();
        layout.partition_row_offsets = partition_row_offsets.data();
        layout.partition_dataset_ids = partition_dataset_ids.data();
        layout.partition_codec_ids = partition_codec_ids.data();
        layout.shard_offsets = shard_offsets.data();
        layout.codecs = &codec;
        layout.num_codecs = 1u;

        cs::shard_storage storage;
        cs::sharded<cs::sparse::blocked_ell> loaded;
        cs::bucketed_blocked_ell_shard optimized_shard;
        cs::bucketed_blocked_ell_partition bucket0;
        cs::bucketed_blocked_ell_partition bucket1;
        cs::init(&storage);
        cs::init(&loaded);
        cs::init(&optimized_shard);
        cs::init(&bucket0);
        cs::init(&bucket1);
        require(cs::create_dataset_blocked_ell_h5(out_path.c_str(), &layout, nullptr, nullptr) != 0, "blocked ell csh5 create failed");
        require(cs::build_bucketed_blocked_ell_partition(&bucket0, blocked.parts[0], 1u, nullptr) != 0,
                "build blocked bucket0 failed");
        require(cs::build_bucketed_blocked_ell_partition(&bucket1, blocked.parts[1], 1u, nullptr) != 0,
                "build blocked bucket1 failed");
        bucket0.exec_to_canonical_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        bucket0.canonical_to_exec_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        bucket1.exec_to_canonical_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        bucket1.canonical_to_exec_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        require(bucket0.exec_to_canonical_cols != nullptr
                    && bucket0.canonical_to_exec_cols != nullptr
                    && bucket1.exec_to_canonical_cols != nullptr
                    && bucket1.canonical_to_exec_cols != nullptr,
                "allocate partition col maps failed");
        optimized_shard.rows = (std::uint32_t) blocked.rows;
        optimized_shard.cols = (std::uint32_t) blocked.cols;
        optimized_shard.nnz = (std::uint64_t) blocked.nnz;
        optimized_shard.partition_count = 2u;
        optimized_shard.partitions = (cs::bucketed_blocked_ell_partition *) std::calloc(2u, sizeof(cs::bucketed_blocked_ell_partition));
        optimized_shard.partition_row_offsets = (std::uint32_t *) std::calloc(3u, sizeof(std::uint32_t));
        optimized_shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        optimized_shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc((std::size_t) blocked.cols, sizeof(std::uint32_t));
        require(optimized_shard.partitions != nullptr
                    && optimized_shard.partition_row_offsets != nullptr
                    && optimized_shard.exec_to_canonical_cols != nullptr
                    && optimized_shard.canonical_to_exec_cols != nullptr,
                "allocate optimized blocked shard failed");
        optimized_shard.partitions[0] = bucket0;
        optimized_shard.partitions[1] = bucket1;
        std::memset(&bucket0, 0, sizeof(bucket0));
        std::memset(&bucket1, 0, sizeof(bucket1));
        optimized_shard.partition_row_offsets[0] = 0u;
        optimized_shard.partition_row_offsets[1] = blocked.parts[0]->rows;
        optimized_shard.partition_row_offsets[2] = blocked.parts[0]->rows + blocked.parts[1]->rows;
        for (std::uint32_t col = 0u; col < blocked.cols; ++col) {
            optimized_shard.partitions[0].exec_to_canonical_cols[col] = col;
            optimized_shard.partitions[0].canonical_to_exec_cols[col] = col;
            optimized_shard.partitions[1].exec_to_canonical_cols[col] = col;
            optimized_shard.partitions[1].canonical_to_exec_cols[col] = col;
            optimized_shard.exec_to_canonical_cols[col] = col;
            optimized_shard.canonical_to_exec_cols[col] = col;
        }
        require(cs::append_bucketed_blocked_ell_shard_h5(out_path.c_str(), 0u, &optimized_shard) != 0,
                "append blocked ell shard failed");
        require(cs::load_header(out_path.c_str(), &loaded, &storage) != 0, "blocked ell csh5 load_header failed");
        require(loaded.num_partitions == blocked.num_partitions, "blocked ell loaded part count mismatch");
        require(loaded.partition_aux[0] == blocked.partition_aux[0], "blocked ell loaded aux mismatch");
        cs::clear(&bucket1);
        cs::clear(&bucket0);
        cs::clear(&optimized_shard);
        cs::clear(&storage);
        cs::clear(&loaded);
        std::remove(out_path.c_str());
    }

    cs::clear(&blocked);
    cs::clear(&src);
    cs::sparse::clear(&local);
    return 0;
}
