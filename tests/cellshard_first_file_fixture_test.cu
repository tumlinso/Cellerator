#include <Cellerator/workbench/dataset_workbench.hh>
#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs = ::cellshard;
namespace wb = ::cellerator::apps::workbench;

namespace {

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    cs::dataset_text_column_view view() const {
        cs::dataset_text_column_view out{};
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? nullptr : offsets.data();
        out.data = data.empty() ? nullptr : data.data();
        return out;
    }
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

owned_text_column make_column(const std::vector<const char *> &values) {
    owned_text_column col;
    std::uint32_t cursor = 0u;
    col.offsets.resize(values.size() + 1u, 0u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const char *value = values[i] != nullptr ? values[i] : "";
        const std::size_t len = std::strlen(value);
        col.offsets[i] = cursor;
        col.data.insert(col.data.end(), value, value + len);
        col.data.push_back(0);
        cursor += (std::uint32_t) len + 1u;
    }
    col.offsets[values.size()] = cursor;
    return col;
}

void fill_blocked_ell_part(cs::sparse::blocked_ell *part,
                           std::uint32_t rows,
                           std::uint32_t cols,
                           std::uint32_t block_size,
                           std::uint32_t ell_cols,
                           const std::uint32_t *block_idx,
                           const float *values,
                           std::size_t block_idx_count,
                           std::size_t value_count) {
    cs::sparse::init(part, rows, cols, (cs::types::nnz_t) value_count, block_size, ell_cols);
    require(cs::sparse::allocate(part) != 0, "blocked ell allocate failed");
    for (std::size_t i = 0; i < block_idx_count; ++i) part->blockColIdx[i] = block_idx[i];
    for (std::size_t i = 0; i < value_count; ++i) part->val[i] = __float2half(values[i]);
}

} // namespace

int main() {
    const std::string out_path = "/tmp/cellshard_first_file_fixture_test.csh5";
    const std::string cache_root = "/tmp/cellshard_first_file_fixture_cache";

    const std::uint32_t block_idx[] = {0u, 1u};
    const float values[] = {1.0f, 7.0f, 2.0f, 3.0f, 8.0f, 4.0f, 5.0f, 6.0f};

    cs::sparse::blocked_ell canonical_part;
    cs::bucketed_blocked_ell_partition bucketed_part;
    cs::bucketed_blocked_ell_partition exec_part;
    cs::bucketed_blocked_ell_shard shard;
    cs::sharded<cs::sparse::blocked_ell> loaded;
    cs::shard_storage storage;

    std::vector<std::uint64_t> partition_rows = {2u};
    std::vector<std::uint64_t> partition_nnz = {8u};
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cs::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = {0u, 2u};
    std::vector<std::uint32_t> partition_dataset_ids = {0u};
    std::vector<std::uint32_t> partition_codec_ids = {0u};
    std::vector<std::uint64_t> shard_offsets = {0u, 2u};
    owned_text_column dataset_ids = make_column({"fixture_counts"});
    owned_text_column matrix_paths = make_column({"/tmp/fixture_counts.mtx"});
    owned_text_column feature_paths = make_column({"/tmp/fixture_features.tsv"});
    owned_text_column barcode_paths = make_column({"/tmp/fixture_barcodes.tsv"});
    owned_text_column metadata_paths = make_column({""});
    owned_text_column global_barcodes = make_column({"bc0", "bc1"});
    owned_text_column feature_ids = make_column({"g0", "g1", "g2", "g3"});
    owned_text_column feature_names = make_column({"Gene0", "Gene1", "Gene2", "Gene3"});
    owned_text_column feature_types = make_column({"gene", "gene", "gene", "gene"});
    std::vector<std::uint32_t> dataset_formats = {2u};
    std::vector<std::uint64_t> dataset_row_begin = {0u};
    std::vector<std::uint64_t> dataset_row_end = {2u};
    std::vector<std::uint64_t> dataset_rows = {2u};
    std::vector<std::uint64_t> dataset_cols = {4u};
    std::vector<std::uint64_t> dataset_nnz = {8u};
    std::vector<std::uint32_t> cell_dataset_ids = {0u, 0u};
    std::vector<std::uint64_t> cell_local_indices = {0u, 1u};
    std::vector<std::uint32_t> feature_dataset_ids = {0u, 0u, 0u, 0u};
    std::vector<std::uint64_t> feature_local_indices = {0u, 1u, 2u, 3u};
    std::vector<std::uint64_t> dataset_feature_offsets = {0u, 4u};
    std::vector<std::uint32_t> dataset_feature_to_global = {0u, 1u, 2u, 3u};

    cs::dataset_codec_descriptor codec{};
    cs::dataset_layout_view layout{};
    cs::dataset_dataset_table_view dataset_view{};
    cs::dataset_provenance_view provenance_view{};
    cs::dataset_execution_view execution{};
    cs::dataset_runtime_service_view runtime_service{};

    std::vector<std::uint32_t> part_formats = {cs::dataset_execution_format_bucketed_blocked_ell};
    std::vector<std::uint32_t> part_block_sizes = {2u};
    std::vector<std::uint32_t> part_bucket_counts = {1u};
    std::vector<float> part_fill_ratios = {1.0f};
    std::vector<std::uint64_t> part_execution_bytes = {128u};
    std::vector<std::uint64_t> part_blocked_ell_bytes = {128u};
    std::vector<std::uint64_t> part_bucketed_blocked_ell_bytes = {128u};
    std::vector<std::uint32_t> shard_formats = {cs::dataset_execution_format_bucketed_blocked_ell};
    std::vector<std::uint32_t> shard_block_sizes = {2u};
    std::vector<std::uint32_t> shard_bucketed_partition_counts = {1u};
    std::vector<std::uint32_t> shard_bucketed_segment_counts = {1u};
    std::vector<float> shard_fill_ratios = {1.0f};
    std::vector<std::uint64_t> shard_execution_bytes = {128u};
    std::vector<std::uint64_t> shard_bucketed_blocked_ell_bytes = {128u};
    std::vector<std::uint32_t> shard_pair_ids = {0u};
    std::vector<std::uint32_t> shard_owner_node_ids = {0u};
    std::vector<std::uint32_t> shard_owner_rank_ids = {0u};

    int rc = 1;

    std::remove(out_path.c_str());
    cs::sparse::init(&canonical_part);
    cs::init(&bucketed_part);
    cs::init(&exec_part);
    cs::init(&shard);
    cs::init(&loaded);
    cs::init(&storage);
    cs::init(&runtime_service);

    try {
        fill_blocked_ell_part(&canonical_part, 2u, 4u, 2u, 4u, block_idx, values, 2u, 8u);
        require(cs::build_bucketed_blocked_ell_partition(&bucketed_part, &canonical_part, 2u, nullptr) != 0,
                "build_bucketed_blocked_ell_partition failed");

        shard.rows = 2u;
        shard.cols = 4u;
        shard.nnz = 8u;
        shard.partition_count = 1u;
        shard.partitions = (cs::bucketed_blocked_ell_partition *) std::calloc(1u, sizeof(cs::bucketed_blocked_ell_partition));
        shard.partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
        shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        require(shard.partitions != nullptr
                    && shard.partition_row_offsets != nullptr
                    && shard.exec_to_canonical_cols != nullptr
                    && shard.canonical_to_exec_cols != nullptr,
                "optimized shard allocation failed");

        shard.partition_row_offsets[0] = 0u;
        shard.partition_row_offsets[1] = 2u;
        shard.exec_to_canonical_cols[0] = 0u;
        shard.exec_to_canonical_cols[1] = 2u;
        shard.exec_to_canonical_cols[2] = 1u;
        shard.exec_to_canonical_cols[3] = 3u;
        shard.canonical_to_exec_cols[0] = 0u;
        shard.canonical_to_exec_cols[1] = 2u;
        shard.canonical_to_exec_cols[2] = 1u;
        shard.canonical_to_exec_cols[3] = 3u;

        shard.partitions[0] = bucketed_part;
        bucketed_part = {};
        require(shard.partitions[0].exec_to_canonical_cols != nullptr
                    && shard.partitions[0].canonical_to_exec_cols != nullptr,
                "bucketed partition column maps missing");
        std::free(shard.partitions[0].exec_to_canonical_cols);
        std::free(shard.partitions[0].canonical_to_exec_cols);
        shard.partitions[0].exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        shard.partitions[0].canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        require(shard.partitions[0].exec_to_canonical_cols != nullptr
                    && shard.partitions[0].canonical_to_exec_cols != nullptr,
                "bucketed partition remap allocation failed");
        std::memcpy(shard.partitions[0].exec_to_canonical_cols, shard.exec_to_canonical_cols, 4u * sizeof(std::uint32_t));
        std::memcpy(shard.partitions[0].canonical_to_exec_cols, shard.canonical_to_exec_cols, 4u * sizeof(std::uint32_t));

        codec.codec_id = 0u;
        codec.family = cs::dataset_codec_family_blocked_ell;
        codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
        codec.scale_value_code = 0u;
        codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
        codec.flags = 0u;

        layout.rows = 2u;
        layout.cols = 4u;
        layout.nnz = 8u;
        layout.num_partitions = 1u;
        layout.num_shards = 1u;
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

        dataset_view.count = 1u;
        dataset_view.dataset_ids = dataset_ids.view();
        dataset_view.matrix_paths = matrix_paths.view();
        dataset_view.feature_paths = feature_paths.view();
        dataset_view.barcode_paths = barcode_paths.view();
        dataset_view.metadata_paths = metadata_paths.view();
        dataset_view.formats = dataset_formats.data();
        dataset_view.row_begin = dataset_row_begin.data();
        dataset_view.row_end = dataset_row_end.data();
        dataset_view.rows = dataset_rows.data();
        dataset_view.cols = dataset_cols.data();
        dataset_view.nnz = dataset_nnz.data();

        provenance_view.global_barcodes = global_barcodes.view();
        provenance_view.cell_dataset_ids = cell_dataset_ids.data();
        provenance_view.cell_local_indices = cell_local_indices.data();
        provenance_view.feature_ids = feature_ids.view();
        provenance_view.feature_names = feature_names.view();
        provenance_view.feature_types = feature_types.view();
        provenance_view.feature_dataset_ids = feature_dataset_ids.data();
        provenance_view.feature_local_indices = feature_local_indices.data();
        provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
        provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

        execution.partition_count = (std::uint32_t) part_formats.size();
        execution.partition_execution_formats = part_formats.data();
        execution.partition_blocked_ell_block_sizes = part_block_sizes.data();
        execution.partition_blocked_ell_bucket_counts = part_bucket_counts.data();
        execution.partition_blocked_ell_fill_ratios = part_fill_ratios.data();
        execution.partition_execution_bytes = part_execution_bytes.data();
        execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.data();
        execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.data();
        execution.shard_count = (std::uint32_t) shard_formats.size();
        execution.shard_execution_formats = shard_formats.data();
        execution.shard_blocked_ell_block_sizes = shard_block_sizes.data();
        execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.data();
        execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.data();
        execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.data();
        execution.shard_execution_bytes = shard_execution_bytes.data();
        execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.data();
        execution.shard_preferred_pair_ids = shard_pair_ids.data();
        execution.shard_owner_node_ids = shard_owner_node_ids.data();
        execution.shard_owner_rank_ids = shard_owner_rank_ids.data();
        execution.preferred_base_format = cs::dataset_execution_format_bucketed_blocked_ell;

        runtime_service.service_mode = cs::dataset_runtime_service_mode_owner_hosted;
        runtime_service.live_write_mode = cs::dataset_live_write_mode_append_only;
        runtime_service.prefer_pack_delivery = 1u;
        runtime_service.remote_pack_delivery = 0u;
        runtime_service.single_reader_coordinator = 1u;
        runtime_service.maintenance_lock_blocks_overwrite = 1u;
        runtime_service.canonical_generation = 1u;
        runtime_service.execution_plan_generation = 1u;
        runtime_service.pack_generation = 1u;
        runtime_service.service_epoch = 1u;
        runtime_service.active_read_generation = 1u;
        runtime_service.staged_write_generation = 1u;

        require(cs::create_dataset_optimized_blocked_ell_h5(out_path.c_str(), &layout, &dataset_view, &provenance_view) != 0,
                "create_dataset_optimized_blocked_ell_h5 failed");
        require(cs::append_bucketed_blocked_ell_shard_h5(out_path.c_str(), 0u, &shard) != 0,
                "append_bucketed_blocked_ell_shard_h5 failed");
        require(cs::append_dataset_execution_h5(out_path.c_str(), &execution) != 0,
                "append_dataset_execution_h5 failed");
        require(cs::append_dataset_runtime_service_h5(out_path.c_str(), &runtime_service) != 0,
                "append_dataset_runtime_service_h5 failed");
        require(cs::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str()) != 0,
                "warm_dataset_blocked_ell_h5_cache failed");
        require(cs::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str()) != 0,
                "warm_dataset_blocked_ell_h5_cache failed");

        wb::dataset_summary summary = wb::summarize_dataset_csh5(out_path);
        require(summary.ok, "summarize_dataset_csh5 failed");
        require(summary.matrix_format == "blocked_ell", "fixture matrix_format mismatch");
        require(summary.payload_layout == "optimized_bucketed_blocked_ell", "fixture payload_layout mismatch");
        require(summary.codecs.size() == 1u, "fixture codec table mismatch");
        require(summary.codecs[0].family == cs::dataset_codec_family_blocked_ell, "fixture codec family mismatch");
        require(summary.preferred_base_format == cs::dataset_execution_format_bucketed_blocked_ell,
                "fixture preferred_base_format mismatch");
        require(summary.partitions.size() == 1u && summary.shards.size() == 1u, "fixture shard summary mismatch");
        require(summary.partitions[0].bucketed_blocked_ell_bytes == 128u, "fixture partition execution bytes mismatch");
        require(summary.shards[0].execution_bytes == 128u, "fixture shard execution bytes mismatch");
        require(summary.runtime_service.available, "fixture runtime service missing");

        require(cs::load_dataset_blocked_ell_h5_header(out_path.c_str(), &loaded, &storage) != 0,
                "load_dataset_blocked_ell_h5_header failed");
        require(cs::bind_dataset_h5_cache(&storage, cache_root.c_str()) != 0,
                "bind_dataset_h5_cache failed");
        require(cs::fetch_dataset_blocked_ell_h5_pack_partition(&exec_part, &loaded, &storage, 0u) != 0,
                "fetch_dataset_blocked_ell_h5_pack_partition failed");
        require(exec_part.cols == 4u, "fixture pack partition cols mismatch");
        require(exec_part.exec_to_canonical_cols != nullptr && exec_part.canonical_to_exec_cols != nullptr,
                "fixture execution column maps missing");
        require(exec_part.exec_to_canonical_cols[0] == 0u
                    && exec_part.exec_to_canonical_cols[1] == 2u
                    && exec_part.exec_to_canonical_cols[2] == 1u
                    && exec_part.exec_to_canonical_cols[3] == 3u,
                "fixture exec_to_canonical remap mismatch");
        require(exec_part.canonical_to_exec_cols[0] == 0u
                    && exec_part.canonical_to_exec_cols[1] == 2u
                    && exec_part.canonical_to_exec_cols[2] == 1u
                    && exec_part.canonical_to_exec_cols[3] == 3u,
                "fixture canonical_to_exec remap mismatch");

        rc = 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "cellShardFirstFileFixtureTest: %s\n", e.what());
    }

    if (storage.backend == cs::shard_storage_backend_dataset_h5) {
        cs::invalidate_dataset_h5_cache(&storage);
    }
    cs::clear(&exec_part);
    cs::clear(&storage);
    cs::clear(&loaded);
    cs::clear(&shard);
    cs::clear(&bucketed_part);
    cs::sparse::clear(&canonical_part);
    std::remove(out_path.c_str());
    return rc;
}
