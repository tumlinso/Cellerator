#include "dataset_workbench.hh"

#include "../compute/preprocess/operators.cuh"
#include "../compute/preprocess/workspace.cuh"
#include "../ingest/common/metadata_table.cuh"
#include "../ingest/h5ad/h5ad_reader.cuh"
#include "../ingest/dataset/dataset_ingest.cuh"

#include "../../extern/CellShard/src/sharded/disk.cuh"
#include "../../extern/CellShard/src/sharded/distributed.cuh"
#include "../../extern/CellShard/src/sharded/sharded_device.cuh"
#include "../../extern/CellShard/src/sharded/sharded_host.cuh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::apps::workbench {

namespace ccommon = ::cellerator::ingest::common;
namespace cpre = ::cellerator::compute::preprocess;
namespace cseries = ::cellerator::ingest::dataset;
namespace cs = ::cellshard;
namespace csd = ::cellshard::distributed;
namespace csv = ::cellshard::device;

namespace {

template<typename T>
class host_buffer {
public:
    host_buffer() = default;

    explicit host_buffer(std::size_t count) {
        resize(count);
    }

    host_buffer(const host_buffer &other) {
        assign_copy(other.data(), other.size());
    }

    host_buffer(host_buffer &&other) noexcept
        : data_(std::move(other.data_)),
          size_(other.size_),
          capacity_(other.capacity_) {
        other.size_ = 0u;
        other.capacity_ = 0u;
    }

    host_buffer &operator=(const host_buffer &other) {
        if (this != &other) assign_copy(other.data(), other.size());
        return *this;
    }

    host_buffer &operator=(host_buffer &&other) noexcept {
        if (this == &other) return *this;
        data_ = std::move(other.data_);
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.size_ = 0u;
        other.capacity_ = 0u;
        return *this;
    }

    void clear() {
        size_ = 0u;
    }

    void reserve(std::size_t capacity) {
        if (capacity <= capacity_) return;
        std::unique_ptr<T[]> next(new T[capacity]);
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (size_ != 0u) std::memcpy(next.get(), data_.get(), size_ * sizeof(T));
        } else {
            for (std::size_t i = 0; i < size_; ++i) next[i] = std::move(data_[i]);
        }
        data_ = std::move(next);
        capacity_ = capacity;
    }

    void resize(std::size_t count) {
        if (count > capacity_) reserve(count);
        size_ = count;
    }

    void assign_copy(const T *src, std::size_t count) {
        resize(count);
        if (count == 0u || src == nullptr) return;
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(data_.get(), src, count * sizeof(T));
        } else {
            for (std::size_t i = 0; i < count; ++i) data_[i] = src[i];
        }
    }

    void assign_fill(std::size_t count, const T &value) {
        resize(count);
        for (std::size_t i = 0; i < count; ++i) data_[i] = value;
    }

    void push_back(const T &value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = value;
    }

    void push_back(T &&value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = std::move(value);
    }

    T *data() {
        return data_.get();
    }

    const T *data() const {
        return data_.get();
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0u;
    }

    T &operator[](std::size_t idx) {
        return data_[idx];
    }

    const T &operator[](std::size_t idx) const {
        return data_[idx];
    }

    T *begin() {
        return data_.get();
    }

    const T *begin() const {
        return data_.get();
    }

    T *end() {
        return data_.get() + size_;
    }

    const T *end() const {
        return data_.get() + size_;
    }

private:
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0u;
    std::size_t capacity_ = 0u;
};

inline void push_issue(std::vector<issue> *issues,
                       issue_severity severity,
                       const std::string &scope,
                       const std::string &message) {
    if (issues == nullptr) return;
    issues->push_back(issue{severity, scope, message});
}

inline bool check_cuda(cudaError_t status,
                       std::vector<issue> *issues,
                       const std::string &scope,
                       const std::string &label) {
    if (status == cudaSuccess) return true;
    push_issue(issues,
               issue_severity::error,
               scope,
               label + ": " + std::string(cudaGetErrorString(status)));
    return false;
}

inline std::string normalized_upper(std::string value) {
    for (char &ch : value) ch = (char) std::toupper((unsigned char) ch);
    return value;
}

inline std::uint32_t to_dataset_execution_format(execution_format format) {
    switch (format) {
        case execution_format::compressed: return cellshard::dataset_execution_format_compressed;
        case execution_format::blocked_ell: return cellshard::dataset_execution_format_blocked_ell;
        case execution_format::bucketed_blocked_ell: return cellshard::dataset_execution_format_bucketed_blocked_ell;
        case execution_format::mixed: return cellshard::dataset_execution_format_mixed;
        case execution_format::unknown: return cellshard::dataset_execution_format_unknown;
    }
    return cellshard::dataset_execution_format_unknown;
}

inline bool append_execution_layout_metadata(const ingest_plan &plan,
                                             std::vector<issue> *issues) {
    std::vector<std::uint32_t> part_formats;
    std::vector<std::uint32_t> part_block_sizes;
    std::vector<std::uint32_t> part_bucket_counts;
    std::vector<float> part_fill_ratios;
    std::vector<std::uint64_t> part_execution_bytes;
    std::vector<std::uint64_t> part_blocked_ell_bytes;
    std::vector<std::uint64_t> part_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> shard_formats;
    std::vector<std::uint32_t> shard_block_sizes;
    std::vector<std::uint32_t> shard_bucketed_partition_counts;
    std::vector<std::uint32_t> shard_bucketed_segment_counts;
    std::vector<float> shard_fill_ratios;
    std::vector<std::uint64_t> shard_execution_bytes;
    std::vector<std::uint64_t> shard_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> shard_pair_ids;
    cellshard::dataset_execution_view execution = {};
    const std::uint32_t persisted_execution_format = cellshard::dataset_execution_format_bucketed_blocked_ell;
    std::uint32_t preferred_base_format = cellshard::dataset_execution_format_unknown;

    part_formats.reserve(plan.parts.size());
    part_block_sizes.reserve(plan.parts.size());
    part_bucket_counts.reserve(plan.parts.size());
    part_fill_ratios.reserve(plan.parts.size());
    part_execution_bytes.reserve(plan.parts.size());
    part_blocked_ell_bytes.reserve(plan.parts.size());
    part_bucketed_blocked_ell_bytes.reserve(plan.parts.size());
    for (const planned_part &part : plan.parts) {
        part_formats.push_back(persisted_execution_format);
        part_block_sizes.push_back(part.blocked_ell_block_size);
        part_bucket_counts.push_back(std::max<std::uint32_t>(1u, part.blocked_ell_bucket_count));
        part_fill_ratios.push_back((float) part.blocked_ell_fill_ratio);
        part_execution_bytes.push_back((std::uint64_t) (part.bucketed_blocked_ell_bytes != 0
            ? part.bucketed_blocked_ell_bytes
            : part.execution_bytes));
        part_blocked_ell_bytes.push_back((std::uint64_t) part.blocked_ell_bytes);
        part_bucketed_blocked_ell_bytes.push_back((std::uint64_t) (part.bucketed_blocked_ell_bytes != 0
            ? part.bucketed_blocked_ell_bytes
            : part.execution_bytes));
    }

    shard_formats.reserve(plan.shards.size());
    shard_block_sizes.reserve(plan.shards.size());
    shard_bucketed_partition_counts.reserve(plan.shards.size());
    shard_bucketed_segment_counts.reserve(plan.shards.size());
    shard_fill_ratios.reserve(plan.shards.size());
    shard_execution_bytes.reserve(plan.shards.size());
    shard_bucketed_blocked_ell_bytes.reserve(plan.shards.size());
    shard_pair_ids.reserve(plan.shards.size());
    for (const planned_shard &shard : plan.shards) {
        const std::uint32_t encoded = persisted_execution_format;
        shard_formats.push_back(encoded);
        shard_block_sizes.push_back(shard.blocked_ell_block_size);
        shard_bucketed_partition_counts.push_back(std::max<std::uint32_t>(1u, shard.bucketed_partition_count));
        shard_bucketed_segment_counts.push_back(std::max<std::uint32_t>(1u, shard.bucketed_segment_count));
        shard_fill_ratios.push_back((float) shard.blocked_ell_fill_ratio);
        shard_execution_bytes.push_back((std::uint64_t) (shard.bucketed_blocked_ell_bytes != 0
            ? shard.bucketed_blocked_ell_bytes
            : shard.execution_bytes));
        shard_bucketed_blocked_ell_bytes.push_back((std::uint64_t) (shard.bucketed_blocked_ell_bytes != 0
            ? shard.bucketed_blocked_ell_bytes
            : shard.execution_bytes));
        shard_pair_ids.push_back(shard.preferred_pair);
        if (preferred_base_format == cellshard::dataset_execution_format_unknown) {
            preferred_base_format = encoded;
        } else if (preferred_base_format != encoded) {
            preferred_base_format = cellshard::dataset_execution_format_mixed;
        }
    }
    if (preferred_base_format == cellshard::dataset_execution_format_unknown) {
        preferred_base_format = cellshard::dataset_execution_format_blocked_ell;
    }

    execution.partition_count = (std::uint32_t) part_formats.size();
    execution.partition_execution_formats = part_formats.empty() ? nullptr : part_formats.data();
    execution.partition_blocked_ell_block_sizes = part_block_sizes.empty() ? nullptr : part_block_sizes.data();
    execution.partition_blocked_ell_bucket_counts = part_bucket_counts.empty() ? nullptr : part_bucket_counts.data();
    execution.partition_blocked_ell_fill_ratios = part_fill_ratios.empty() ? nullptr : part_fill_ratios.data();
    execution.partition_execution_bytes = part_execution_bytes.empty() ? nullptr : part_execution_bytes.data();
    execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.empty() ? nullptr : part_blocked_ell_bytes.data();
    execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.empty() ? nullptr : part_bucketed_blocked_ell_bytes.data();
    execution.shard_count = (std::uint32_t) shard_formats.size();
    execution.shard_execution_formats = shard_formats.empty() ? nullptr : shard_formats.data();
    execution.shard_blocked_ell_block_sizes = shard_block_sizes.empty() ? nullptr : shard_block_sizes.data();
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.empty() ? nullptr : shard_bucketed_partition_counts.data();
    execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.empty() ? nullptr : shard_bucketed_segment_counts.data();
    execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.empty() ? nullptr : shard_fill_ratios.data();
    execution.shard_execution_bytes = shard_execution_bytes.empty() ? nullptr : shard_execution_bytes.data();
    execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.empty() ? nullptr : shard_bucketed_blocked_ell_bytes.data();
    execution.shard_preferred_pair_ids = shard_pair_ids.empty() ? nullptr : shard_pair_ids.data();
    execution.preferred_base_format = preferred_base_format;

    if (!cellshard::append_dataset_execution_h5(plan.policy.output_path.c_str(), &execution)) {
        push_issue(issues, issue_severity::error, "execution", "failed to append execution layout metadata");
        return false;
    }
    {
        cellshard::dataset_runtime_service_view runtime_service{};
        cellshard::init(&runtime_service);
        runtime_service.service_mode = cellshard::dataset_runtime_service_mode_owner_hosted;
        runtime_service.live_write_mode = cellshard::dataset_live_write_mode_append_only;
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
        if (!cellshard::append_dataset_runtime_service_h5(plan.policy.output_path.c_str(), &runtime_service)) {
            push_issue(issues, issue_severity::error, "runtime_service", "failed to append runtime service metadata");
            return false;
        }
    }
    return true;
}

inline host_buffer<unsigned char> build_gene_flags(const dataset_summary &summary,
                                                   const preprocess_config &config) {
    host_buffer<unsigned char> flags;
    flags.assign_fill(summary.cols, static_cast<unsigned char>(0u));
    if (!config.mark_mito_from_feature_names || config.mito_prefix.empty()) return flags;
    const std::string prefix = normalized_upper(config.mito_prefix);
    for (std::size_t i = 0; i < summary.feature_names.size() && i < flags.size(); ++i) {
        std::string name = normalized_upper(summary.feature_names[i]);
        if (name.rfind(prefix, 0) == 0) flags[i] = (unsigned char) cpre::gene_flag_mito;
    }
    return flags;
}

template<typename MatrixT>
inline unsigned int max_partition_rows(const cs::sharded<MatrixT> &view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view.num_partitions; ++part) best = std::max(best, view.partition_rows[part]);
    return (unsigned int) best;
}

template<typename MatrixT>
inline unsigned int max_partition_nnz(const cs::sharded<MatrixT> &view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view.num_partitions; ++part) best = std::max(best, view.partition_nnz[part]);
    return (unsigned int) best;
}

inline int bind_uploaded_part_view(csv::blocked_ell_view *out,
                                   const cs::sharded<cs::sparse::blocked_ell> *host,
                                   const csv::partition_record<cs::sparse::blocked_ell> *record,
                                   unsigned long part_id) {
    if (out == nullptr || host == nullptr || record == nullptr) return 0;
    if (part_id >= host->num_partitions) return 0;
    if (record->a0 == nullptr && host->partition_rows[part_id] != 0) return 0;
    if (record->a1 == nullptr && host->partition_rows[part_id] != 0) return 0;
    out->rows = (unsigned int) host->partition_rows[part_id];
    out->cols = (unsigned int) host->cols;
    out->nnz = (unsigned int) host->partition_nnz[part_id];
    out->block_size = cs::sparse::unpack_blocked_ell_block_size(host->partition_aux[part_id]);
    out->ell_cols = cs::sparse::unpack_blocked_ell_cols(host->partition_aux[part_id]);
    out->blockColIdx = (unsigned int *) record->a0;
    out->val = (__half *) record->a1;
    return 1;
}

struct single_part_blocked_ell_host {
    cs::sharded<cs::sparse::blocked_ell> view;
    cs::sparse::blocked_ell *parts[1];
    unsigned long partition_offsets[2];
    unsigned long partition_rows[1];
    unsigned long partition_nnz[1];
    unsigned long partition_aux[1];
    unsigned long shard_offsets[2];
    unsigned long shard_parts[2];

    explicit single_part_blocked_ell_host(cs::sparse::blocked_ell *part) {
        cs::init(&view);
        parts[0] = part;
        partition_offsets[0] = 0ul;
        partition_offsets[1] = part != nullptr ? (unsigned long) part->rows : 0ul;
        partition_rows[0] = part != nullptr ? (unsigned long) part->rows : 0ul;
        partition_nnz[0] = part != nullptr ? (unsigned long) part->nnz : 0ul;
        partition_aux[0] = part != nullptr ? cs::partition_aux(part) : 0ul;
        shard_offsets[0] = 0ul;
        shard_offsets[1] = partition_offsets[1];
        shard_parts[0] = 0ul;
        shard_parts[1] = 1ul;
        view.rows = partition_offsets[1];
        view.cols = part != nullptr ? (unsigned long) part->cols : 0ul;
        view.nnz = part != nullptr ? (unsigned long) part->nnz : 0ul;
        view.num_partitions = 1ul;
        view.partition_capacity = 1ul;
        view.parts = parts;
        view.partition_offsets = partition_offsets;
        view.partition_rows = partition_rows;
        view.partition_nnz = partition_nnz;
        view.partition_aux = partition_aux;
        view.num_shards = 1ul;
        view.shard_capacity = 1ul;
        view.shard_offsets = shard_offsets;
        view.shard_parts = shard_parts;
    }
};

struct owned_sliced_ell_host {
    cs::sparse::sliced_ell part;

    owned_sliced_ell_host() {
        cs::sparse::init(&part);
    }

    ~owned_sliced_ell_host() {
        cs::sparse::clear(&part);
    }

    owned_sliced_ell_host(const owned_sliced_ell_host &) = delete;
    owned_sliced_ell_host &operator=(const owned_sliced_ell_host &) = delete;
};

inline bool build_preprocess_sliced_segment(const cs::sparse::blocked_ell *src,
                                            std::uint32_t slice_rows,
                                            cs::sparse::sliced_ell *out) {
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> slice_widths;
    const std::uint32_t rows = src != nullptr ? src->rows : 0u;
    const std::uint32_t block_size = src != nullptr ? src->block_size : 0u;
    const std::uint32_t ell_width_blocks = block_size != 0u ? cs::sparse::ell_width_blocks(src) : 0u;
    std::uint32_t row = 0u;

    if (src == nullptr || out == nullptr || block_size == 0u || slice_rows == 0u) return false;
    cs::sparse::clear(out);
    cs::sparse::init(out, src->rows, src->cols, src->nnz);

    row_offsets.push_back(0u);
    while (row < rows) {
        const std::uint32_t row_end = std::min<std::uint32_t>(rows, row + slice_rows);
        std::uint32_t max_width = 0u;
        for (std::uint32_t local_row = row; local_row < row_end; ++local_row) {
            const std::uint32_t row_block = local_row / block_size;
            std::uint32_t row_nnz = 0u;
            for (std::uint32_t ell_col = 0u; ell_col < src->ell_cols; ++ell_col) {
                const std::uint32_t slot = ell_col / block_size;
                const std::uint32_t lane_in_block = ell_col % block_size;
                const std::uint32_t block_col = src->blockColIdx[(unsigned long) row_block * ell_width_blocks + slot];
                const std::uint32_t col = block_col != cs::sparse::blocked_ell_invalid_col
                    ? block_col * block_size + lane_in_block
                    : src->cols;
                const float value = __half2float(src->val[(unsigned long) local_row * src->ell_cols + ell_col]);
                if (col < src->cols && value != 0.0f) ++row_nnz;
            }
            max_width = std::max(max_width, row_nnz);
        }
        slice_widths.push_back(max_width);
        row_offsets.push_back(row_end);
        row = row_end;
    }

    if (!cs::sparse::allocate(out,
                              (std::uint32_t) slice_widths.size(),
                              row_offsets.data(),
                              slice_widths.data())) {
        cs::sparse::clear(out);
        return false;
    }

    for (std::uint32_t slice = 0u; slice < out->slice_count; ++slice) {
        const std::uint32_t row_begin = out->slice_row_offsets[slice];
        const std::uint32_t row_end = out->slice_row_offsets[slice + 1u];
        const std::uint32_t width = out->slice_widths[slice];
        const std::size_t slice_base = cs::sparse::slice_slot_base(out, slice);
        for (std::uint32_t local_row = row_begin; local_row < row_end; ++local_row) {
            const std::uint32_t row_block = local_row / block_size;
            const std::size_t dst_row_base = slice_base + (std::size_t) (local_row - row_begin) * (std::size_t) width;
            std::uint32_t dst_slot = 0u;
            for (std::uint32_t ell_col = 0u; ell_col < src->ell_cols; ++ell_col) {
                const std::uint32_t slot = ell_col / block_size;
                const std::uint32_t lane_in_block = ell_col % block_size;
                const std::uint32_t block_col = src->blockColIdx[(unsigned long) row_block * ell_width_blocks + slot];
                const std::uint32_t col = block_col != cs::sparse::blocked_ell_invalid_col
                    ? block_col * block_size + lane_in_block
                    : src->cols;
                const __half value = src->val[(unsigned long) local_row * src->ell_cols + ell_col];
                if (col >= src->cols || __half2float(value) == 0.0f) continue;
                out->col_idx[dst_row_base + dst_slot] = col;
                out->val[dst_row_base + dst_slot] = value;
                ++dst_slot;
            }
        }
    }

    return true;
}

inline bool fetch_execution_partition(cellshard::bucketed_blocked_ell_partition *out,
                                      cs::sharded<cs::sparse::blocked_ell> *matrix,
                                      cs::shard_storage *storage,
                                      unsigned long part_id,
                                      std::vector<issue> *issues,
                                      const std::string &scope,
                                      const std::string &label) {
    if (!cs::fetch_dataset_blocked_ell_h5_execution_partition(out, matrix, storage, part_id)) {
        push_issue(issues, issue_severity::error, scope, "failed to fetch bucketed execution partition for " + label);
        return false;
    }
    return true;
}

inline cellshard::dataset_text_column_view as_text_view(const ccommon::text_column *column) {
    cellshard::dataset_text_column_view view;
    view.count = column != nullptr ? column->count : 0u;
    view.bytes = column != nullptr ? column->bytes : 0u;
    view.offsets = column != nullptr ? column->offsets : nullptr;
    view.data = column != nullptr ? column->data : nullptr;
    return view;
}

struct owned_metadata_table {
    ccommon::metadata_table table;

    owned_metadata_table() { ccommon::init(&table); }
    ~owned_metadata_table() { ccommon::clear(&table); }
    owned_metadata_table(const owned_metadata_table &) = delete;
    owned_metadata_table &operator=(const owned_metadata_table &) = delete;
};

struct owned_observation_text_column {
    ccommon::text_column values;

    owned_observation_text_column() { ccommon::init(&values); }
    ~owned_observation_text_column() { ccommon::clear(&values); }
    owned_observation_text_column(const owned_observation_text_column &) = delete;
    owned_observation_text_column &operator=(const owned_observation_text_column &) = delete;
};

struct owned_observation_metadata_column {
    std::string name;
    std::uint32_t type = cs::dataset_observation_metadata_type_none;
    std::unique_ptr<owned_observation_text_column> text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;

    cs::dataset_observation_metadata_column_view view() const {
        cs::dataset_observation_metadata_column_view out{};
        out.name = name.c_str();
        out.type = type;
        out.text_values = text_values != nullptr ? as_text_view(&text_values->values) : cs::dataset_text_column_view{};
        out.float32_values = float32_values.empty() ? nullptr : float32_values.data();
        out.uint8_values = uint8_values.empty() ? nullptr : uint8_values.data();
        return out;
    }
};

struct normalized_day_value {
    std::string label;
    float numeric = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t is_postnatal = 0u;
    bool has_numeric = false;
};

struct loaded_observation_metadata {
    std::unique_ptr<owned_metadata_table> table;
    std::vector<std::string> column_names;
    std::vector<int> global_text_to_local;
    int day_column = -1;
};

inline bool load_source_metadata_table(const source_entry &source,
                                       owned_metadata_table *owned,
                                       std::vector<issue> *issues) {
    std::string error;
    if (owned == nullptr) return false;
    if (source.format == cseries::source_h5ad) {
        if (!cellerator::ingest::h5ad::load_metadata_table(source.matrix_path.c_str(), &owned->table, &error)) {
            push_issue(issues,
                       issue_severity::warning,
                       "metadata",
                       error.empty() ? ("failed to load h5ad metadata for " + source.dataset_id) : error);
            return false;
        }
        return true;
    }
    if (source.metadata_path.empty()) return false;
    if (!ccommon::load_tsv(source.metadata_path.c_str(), &owned->table, 1)) {
        push_issue(issues, issue_severity::warning, "metadata", "failed to load metadata table for " + source.dataset_id);
        return false;
    }
    return true;
}

inline std::string trim_copy(std::string value) {
    std::size_t begin = 0u;
    std::size_t end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1u])) --end;
    return value.substr(begin, end - begin);
}

inline std::string lower_copy(std::string value) {
    for (char &ch : value) ch = (char) std::tolower((unsigned char) ch);
    return value;
}

inline std::string sanitize_metadata_column_name(const char *raw_name,
                                                 unsigned int column_index) {
    std::string name = trim_copy(raw_name != nullptr ? raw_name : "");
    if (!name.empty()) return name;
    if (column_index == 0u) return "row_id";
    return "column_" + std::to_string(column_index);
}

inline std::vector<std::string> make_unique_metadata_column_names(const ccommon::metadata_table &table) {
    std::unordered_map<std::string, unsigned int> counts;
    std::vector<std::string> names;
    names.reserve(table.num_cols);
    for (unsigned int col = 0; col < table.num_cols; ++col) {
        const std::string base = sanitize_metadata_column_name(ccommon::column_name(&table, col), col);
        const unsigned int seen = counts[base]++;
        if (seen == 0u) names.push_back(base);
        else names.push_back(base + "_" + std::to_string(seen + 1u));
    }
    return names;
}

inline int find_day_column_index(const std::vector<std::string> &column_names) {
    int stage_index = -1;
    for (std::size_t i = 0; i < column_names.size(); ++i) {
        const std::string name = lower_copy(column_names[i]);
        if (name == "day" || name == "embryonic_day_label") return (int) i;
        if (stage_index < 0 && name == "stage") stage_index = (int) i;
    }
    return stage_index;
}

inline normalized_day_value normalize_day_value(const char *raw_value) {
    normalized_day_value out;
    const std::string value = trim_copy(raw_value != nullptr ? raw_value : "");
    if (value.empty()) return out;

    out.label = value;
    if (value == "P0" || value == "p0") {
        out.is_postnatal = 1u;
        return out;
    }

    if (value.size() > 1u && (value[0] == 'E' || value[0] == 'e')) {
        char *end = nullptr;
        const float parsed = std::strtof(value.c_str() + 1u, &end);
        if (end != value.c_str() + 1u && end != nullptr && *end == '\0' && std::isfinite(parsed)) {
            out.numeric = parsed;
            out.has_numeric = true;
        }
    }
    return out;
}

struct browse_cache_owned {
    host_buffer<std::uint32_t> selected_feature_indices;
    host_buffer<float> gene_sum;
    host_buffer<float> gene_detected;
    host_buffer<float> gene_sq_sum;
    host_buffer<float> dataset_feature_mean;
    host_buffer<float> shard_feature_mean;
    host_buffer<std::uint32_t> partition_sample_row_offsets;
    host_buffer<std::uint64_t> partition_sample_global_rows;
    host_buffer<float> partition_sample_values;
};

struct gene_metric_partial {
    host_buffer<float> gene_sum;
    host_buffer<float> gene_detected;
    host_buffer<float> gene_sq_sum;
    float active_rows = 0.0f;
    bool ok = false;
};

struct selected_feature_partial {
    host_buffer<float> dataset_feature_sum;
    host_buffer<float> shard_feature_sum;
    bool ok = false;
};

#include "dataset_workbench_cuda/accumulate_selected_feature_sums_kernel.cuh"
#include "dataset_workbench_cuda/extract_sample_tile_kernel.cuh"
#include "dataset_workbench_cuda/accumulate_gene_metrics_blocked_ell_kernel.cuh"
#include "dataset_workbench_cuda/accumulate_selected_feature_sums_blocked_ell_kernel.cuh"
#include "dataset_workbench_cuda/extract_sample_tile_blocked_ell_kernel.cuh"

inline bool append_embedded_metadata_tables(const ingest_plan &plan,
                                            std::vector<issue> *issues) {
    std::vector<owned_metadata_table *> owned;
    std::vector<cellshard::dataset_metadata_table_view> views;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> global_row_begin;
    std::vector<std::uint64_t> global_row_end;
    cellshard::dataset_embedded_metadata_view metadata_view{};
    bool ok = true;

    owned.reserve(plan.datasets.size());
    views.reserve(plan.datasets.size());
    dataset_indices.reserve(plan.datasets.size());
    global_row_begin.reserve(plan.datasets.size());
    global_row_end.reserve(plan.datasets.size());

    for (std::size_t i = 0; i < plan.datasets.size(); ++i) {
        const planned_dataset &dataset = plan.datasets[i];
        const source_entry &source = plan.sources[dataset.source_index];
        if (source.format != cseries::source_h5ad && source.metadata_path.empty()) continue;

        owned_metadata_table *owned_table = new owned_metadata_table();
        if (!load_source_metadata_table(source, owned_table, issues)) {
            delete owned_table;
            continue;
        }
        if (owned_table->table.num_rows != dataset.rows) {
            push_issue(issues,
                       issue_severity::error,
                       "metadata",
                       "metadata row count does not match barcodes for " + dataset.dataset_id);
            delete owned_table;
            ok = false;
            continue;
        }

        dataset_indices.push_back((std::uint32_t) i);
        global_row_begin.push_back((std::uint64_t) dataset.global_row_begin);
        global_row_end.push_back((std::uint64_t) dataset.global_row_end);
        views.push_back(cellshard::dataset_metadata_table_view{
            owned_table->table.num_rows,
            owned_table->table.num_cols,
            as_text_view(&owned_table->table.column_names),
            as_text_view(&owned_table->table.field_values),
            owned_table->table.row_offsets
        });
        owned.push_back(owned_table);
    }

    if (!ok) {
        for (owned_metadata_table *table : owned) delete table;
        return false;
    }

    metadata_view.count = (std::uint32_t) views.size();
    metadata_view.dataset_indices = dataset_indices.empty() ? nullptr : dataset_indices.data();
    metadata_view.global_row_begin = global_row_begin.empty() ? nullptr : global_row_begin.data();
    metadata_view.global_row_end = global_row_end.empty() ? nullptr : global_row_end.data();
    metadata_view.tables = views.empty() ? nullptr : views.data();

    if (!cellshard::append_dataset_embedded_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append embedded metadata to dataset.csh5");
        ok = false;
    }

    for (owned_metadata_table *table : owned) delete table;
    return ok;
}

inline bool append_observation_metadata_table(const ingest_plan &plan,
                                             std::vector<issue> *issues) {
    std::vector<loaded_observation_metadata> dataset_metadata(plan.datasets.size());
    std::vector<std::string> text_column_names;
    std::unordered_map<std::string, std::size_t> text_column_indices;
    std::vector<std::unique_ptr<owned_observation_metadata_column>> columns;
    std::vector<cs::dataset_observation_metadata_column_view> views;
    cs::dataset_observation_metadata_view metadata_view{};
    bool saw_any_metadata = false;
    bool need_day_columns = false;

    if (plan.total_rows > (unsigned long) std::numeric_limits<std::uint32_t>::max()) {
        push_issue(issues, issue_severity::error, "metadata", "typed observation metadata currently requires <= 2^32-1 rows");
        return false;
    }

    for (std::size_t i = 0; i < plan.datasets.size(); ++i) {
        const planned_dataset &dataset = plan.datasets[i];
        const source_entry &source = plan.sources[dataset.source_index];
        if (source.format != cseries::source_h5ad && source.metadata_path.empty()) continue;

        auto owned = std::make_unique<owned_metadata_table>();
        if (!load_source_metadata_table(source, owned.get(), issues)) {
            continue;
        }
        if (owned->table.num_rows != dataset.rows) {
            push_issue(issues,
                       issue_severity::error,
                       "metadata",
                       "metadata row count does not match barcodes for " + dataset.dataset_id);
            return false;
        }

        loaded_observation_metadata loaded;
        loaded.column_names = make_unique_metadata_column_names(owned->table);
        loaded.day_column = find_day_column_index(loaded.column_names);
        loaded.table = std::move(owned);
        if (loaded.day_column >= 0) need_day_columns = true;
        for (const std::string &name : loaded.column_names) {
            if (text_column_indices.find(name) != text_column_indices.end()) continue;
            text_column_indices.emplace(name, text_column_names.size());
            text_column_names.push_back(name);
        }
        dataset_metadata[i] = std::move(loaded);
        saw_any_metadata = true;
    }

    if (!saw_any_metadata) return true;

    for (loaded_observation_metadata &loaded : dataset_metadata) {
        loaded.global_text_to_local.assign(text_column_names.size(), -1);
        for (std::size_t local = 0; local < loaded.column_names.size(); ++local) {
            const auto it = text_column_indices.find(loaded.column_names[local]);
            if (it != text_column_indices.end()) loaded.global_text_to_local[it->second] = (int) local;
        }
    }

    columns.reserve(text_column_names.size() + (need_day_columns ? 3u : 0u));
    for (const std::string &name : text_column_names) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_text;
        column->text_values = std::make_unique<owned_observation_text_column>();
        columns.push_back(std::move(column));
    }

    owned_observation_metadata_column *day_label_column = nullptr;
    owned_observation_metadata_column *day_numeric_column = nullptr;
    owned_observation_metadata_column *postnatal_column = nullptr;
    if (need_day_columns) {
        auto label = std::make_unique<owned_observation_metadata_column>();
        label->name = "embryonic_day_label";
        label->type = cs::dataset_observation_metadata_type_text;
        label->text_values = std::make_unique<owned_observation_text_column>();
        day_label_column = label.get();
        columns.push_back(std::move(label));

        auto numeric = std::make_unique<owned_observation_metadata_column>();
        numeric->name = "embryonic_day";
        numeric->type = cs::dataset_observation_metadata_type_float32;
        numeric->float32_values.reserve(plan.total_rows);
        day_numeric_column = numeric.get();
        columns.push_back(std::move(numeric));

        auto postnatal = std::make_unique<owned_observation_metadata_column>();
        postnatal->name = "is_postnatal";
        postnatal->type = cs::dataset_observation_metadata_type_uint8;
        postnatal->uint8_values.reserve(plan.total_rows);
        postnatal_column = postnatal.get();
        columns.push_back(std::move(postnatal));
    }

    for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
        const planned_dataset &dataset = plan.datasets[dataset_index];
        const loaded_observation_metadata &loaded = dataset_metadata[dataset_index];
        for (unsigned long row = 0; row < dataset.rows; ++row) {
            for (std::size_t global_col = 0; global_col < text_column_names.size(); ++global_col) {
                const int local_col = global_col < loaded.global_text_to_local.size()
                    ? loaded.global_text_to_local[global_col]
                    : -1;
                const char *value = (loaded.table != nullptr && local_col >= 0)
                    ? ccommon::field(&loaded.table->table, (unsigned int) row, (unsigned int) local_col)
                    : "";
                if (!ccommon::append(&columns[global_col]->text_values->values,
                                     value != nullptr ? value : "",
                                     std::strlen(value != nullptr ? value : ""))) {
                    push_issue(issues, issue_severity::error, "metadata", "failed to build observation metadata text column");
                    return false;
                }
            }

            if (need_day_columns) {
                const char *raw_day = (loaded.table != nullptr && loaded.day_column >= 0)
                    ? ccommon::field(&loaded.table->table, (unsigned int) row, (unsigned int) loaded.day_column)
                    : "";
                const normalized_day_value normalized = normalize_day_value(raw_day);
                if (!ccommon::append(&day_label_column->text_values->values,
                                     normalized.label.c_str(),
                                     normalized.label.size())) {
                    push_issue(issues, issue_severity::error, "metadata", "failed to build embryonic_day_label column");
                    return false;
                }
                day_numeric_column->float32_values.push_back(
                    normalized.has_numeric ? normalized.numeric : std::numeric_limits<float>::quiet_NaN());
                postnatal_column->uint8_values.push_back(normalized.is_postnatal);
            }
        }
    }

    views.reserve(columns.size());
    for (const std::unique_ptr<owned_observation_metadata_column> &column : columns) {
        views.push_back(column->view());
    }

    metadata_view.rows = plan.total_rows;
    metadata_view.cols = (std::uint32_t) views.size();
    metadata_view.columns = views.empty() ? nullptr : views.data();

    if (!cs::append_dataset_observation_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append typed observation metadata to dataset.csh5");
        return false;
    }

    return true;
}

inline host_buffer<unsigned int> choose_sample_rows(unsigned int rows, unsigned int sample_rows) {
    host_buffer<unsigned int> out;
    out.assign_fill(sample_rows, std::numeric_limits<unsigned int>::max());
    if (rows == 0 || sample_rows == 0) return out;
    if (rows <= sample_rows) {
        for (unsigned int i = 0; i < rows; ++i) out[i] = i;
        return out;
    }
    for (unsigned int i = 0; i < sample_rows; ++i) {
        const unsigned long num = (unsigned long) i * (unsigned long) rows;
        out[i] = (unsigned int) std::min<unsigned long>(rows - 1u, num / sample_rows);
    }
    return out;
}

[[maybe_unused]] inline bool build_gene_metric_partials(const std::string &path,
                                                        const host_buffer<int> &shard_owner,
                                                        unsigned int worker_slot,
                                                        int device_id,
                                                        unsigned int cols,
                                                        unsigned int max_rows,
                                                        unsigned int max_nnz,
                                                        gene_metric_partial *out,
                                                        std::vector<issue> *issues) {
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::compressed> device_state;
    cpre::device_workspace workspace;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);
    cpre::init(&workspace);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !csv::reserve(&device_state, matrix.num_partitions)
        || !cpre::setup(&workspace, device_id, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_rows, cols, max_nnz)
        || !cpre::zero_gene_metrics(&workspace, cols)) {
        push_issue(issues, issue_severity::error, "browse", "failed to set up multi-GPU gene metric worker");
        goto done;
    }

    for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
        const unsigned long part_begin = cs::first_partition_in_shard(&matrix, shard_id);
        const unsigned long part_end = cs::last_partition_in_shard(&matrix, shard_id);
        for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
            csv::compressed_view part_view;
            if (!cs::fetch_partition(&matrix, &storage, part_id)
                || !check_cuda(csv::upload_partition(&device_state, &matrix, part_id, device_id), issues, "browse", "upload_partition gene metrics")
                || !cpre::bind_uploaded_part_view(&part_view, &matrix, device_state.parts + part_id, part_id)
                || !cpre::accumulate_gene_metrics(&part_view, &workspace, nullptr, nullptr)
                || !check_cuda(cudaStreamSynchronize(workspace.stream), issues, "browse", "cudaStreamSynchronize gene metrics")
                || !check_cuda(csv::release_partition(&device_state, part_id), issues, "browse", "release_partition gene metrics")) {
                goto done;
            }
            cs::drop_partition(&matrix, part_id);
        }
    }

    out->gene_sum.assign_fill(cols, 0.0f);
    out->gene_detected.assign_fill(cols, 0.0f);
    out->gene_sq_sum.assign_fill(cols, 0.0f);
    if (!check_cuda(cudaMemcpy(out->gene_sum.data(), workspace.d_gene_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                    issues, "browse", "cudaMemcpy gene_sum")
        || !check_cuda(cudaMemcpy(out->gene_detected.data(), workspace.d_gene_detected, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy gene_detected")
        || !check_cuda(cudaMemcpy(out->gene_sq_sum.data(), workspace.d_gene_sq_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy gene_sq_sum")
        || !check_cuda(cudaMemcpy(&out->active_rows, workspace.d_active_rows, sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy active_rows")) {
        goto done;
    }
    out->ok = true;

done:
    cpre::clear(&workspace);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return out->ok;
}

inline bool build_gene_metric_partials_blocked_ell(const std::string &path,
                                                   const std::string &cache_dir,
                                                   const host_buffer<int> &shard_owner,
                                                   unsigned int worker_slot,
                                                   int device_id,
                                                   unsigned int cols,
                                                   gene_metric_partial *out,
                                                   std::vector<issue> *issues) {
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::blocked_ell> device_state;
    cellshard::bucketed_blocked_ell_partition exec_part;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);
    cellshard::init(&exec_part);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !cs::bind_dataset_h5_cache(&storage, cache_dir.c_str())
        || !csv::reserve(&device_state, 1ul)
        || !check_cuda(cudaSetDevice(device_id), issues, "browse", "cudaSetDevice blocked_ell gene metrics")) {
        push_issue(issues, issue_severity::error, "browse", "failed to set up blocked-ell gene metric worker");
        goto done;
    }

    out->gene_sum.assign_fill(cols, 0.0f);
    out->gene_detected.assign_fill(cols, 0.0f);
    out->gene_sq_sum.assign_fill(cols, 0.0f);

    {
        float *d_gene_sum = nullptr;
        float *d_gene_detected = nullptr;
        float *d_gene_sq_sum = nullptr;
        host_buffer<float> host_exec_gene_sum;
        host_buffer<float> host_exec_gene_detected;
        host_buffer<float> host_exec_gene_sq_sum;

        if (!check_cuda(cudaMalloc((void **) &d_gene_sum, (std::size_t) cols * sizeof(float)),
                        issues, "browse", "cudaMalloc blocked_ell gene_sum")
            || !check_cuda(cudaMalloc((void **) &d_gene_detected, (std::size_t) cols * sizeof(float)),
                           issues, "browse", "cudaMalloc blocked_ell gene_detected")
            || !check_cuda(cudaMalloc((void **) &d_gene_sq_sum, (std::size_t) cols * sizeof(float)),
                           issues, "browse", "cudaMalloc blocked_ell gene_sq_sum")
            || !check_cuda(cudaMemset(d_gene_sum, 0, (std::size_t) cols * sizeof(float)),
                           issues, "browse", "cudaMemset blocked_ell gene_sum")
            || !check_cuda(cudaMemset(d_gene_detected, 0, (std::size_t) cols * sizeof(float)),
                           issues, "browse", "cudaMemset blocked_ell gene_detected")
            || !check_cuda(cudaMemset(d_gene_sq_sum, 0, (std::size_t) cols * sizeof(float)),
                           issues, "browse", "cudaMemset blocked_ell gene_sq_sum")) {
            if (d_gene_sq_sum != nullptr) cudaFree(d_gene_sq_sum);
            if (d_gene_detected != nullptr) cudaFree(d_gene_detected);
            if (d_gene_sum != nullptr) cudaFree(d_gene_sum);
            goto done;
        }

        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
            const unsigned long part_begin = cs::first_partition_in_shard(&matrix, shard_id);
            const unsigned long part_end = cs::last_partition_in_shard(&matrix, shard_id);
            for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
                if (!fetch_execution_partition(&exec_part, &matrix, &storage, part_id, issues, "browse", "blocked_ell gene metrics")) {
                    if (d_gene_sq_sum != nullptr) cudaFree(d_gene_sq_sum);
                    if (d_gene_detected != nullptr) cudaFree(d_gene_detected);
                    if (d_gene_sum != nullptr) cudaFree(d_gene_sum);
                    goto done;
                }
                if (!check_cuda(cudaMemset(d_gene_sum, 0, (std::size_t) cols * sizeof(float)),
                                issues, "browse", "cudaMemset blocked_ell part gene_sum")
                    || !check_cuda(cudaMemset(d_gene_detected, 0, (std::size_t) cols * sizeof(float)),
                                   issues, "browse", "cudaMemset blocked_ell part gene_detected")
                    || !check_cuda(cudaMemset(d_gene_sq_sum, 0, (std::size_t) cols * sizeof(float)),
                                   issues, "browse", "cudaMemset blocked_ell part gene_sq_sum")) {
                    cudaFree(d_gene_sq_sum);
                    cudaFree(d_gene_detected);
                    cudaFree(d_gene_sum);
                    goto done;
                }
                for (std::uint32_t segment = 0u; segment < exec_part.segment_count; ++segment) {
                    csv::blocked_ell_view part_view;
                    single_part_blocked_ell_host host(exec_part.segments + segment);
                    const unsigned long total = (unsigned long) exec_part.segments[segment].rows * (unsigned long) exec_part.segments[segment].ell_cols;
                    const unsigned int blocks = (unsigned int) std::max<unsigned long>(1ul, std::min<unsigned long>(4096ul, (total + 255ul) / 256ul));
                    if (!check_cuda(csv::upload_partition(&device_state, &host.view, 0ul, device_id),
                                    issues, "browse", "upload_execution_segment blocked_ell gene metrics")
                        || !bind_uploaded_part_view(&part_view, &host.view, device_state.parts, 0ul)) {
                        if (d_gene_sq_sum != nullptr) cudaFree(d_gene_sq_sum);
                        if (d_gene_detected != nullptr) cudaFree(d_gene_detected);
                        if (d_gene_sum != nullptr) cudaFree(d_gene_sum);
                        goto done;
                    }
                    accumulate_gene_metrics_blocked_ell_kernel<<<blocks, 256>>>(
                        part_view,
                        d_gene_sum,
                        d_gene_detected,
                        d_gene_sq_sum
                    );
                    if (!check_cuda(cudaGetLastError(), issues, "browse", "accumulate_gene_metrics_blocked_ell_kernel")
                        || !check_cuda(cudaDeviceSynchronize(), issues, "browse", "cudaDeviceSynchronize blocked_ell gene metrics")
                        || !check_cuda(csv::release_partition(&device_state, 0ul),
                                       issues, "browse", "release_execution_segment blocked_ell gene metrics")) {
                        if (d_gene_sq_sum != nullptr) cudaFree(d_gene_sq_sum);
                        if (d_gene_detected != nullptr) cudaFree(d_gene_detected);
                        if (d_gene_sum != nullptr) cudaFree(d_gene_sum);
                        goto done;
                    }
                    out->active_rows += (float) exec_part.segments[segment].rows;
                }
                host_exec_gene_sum.assign_fill(cols, 0.0f);
                host_exec_gene_detected.assign_fill(cols, 0.0f);
                host_exec_gene_sq_sum.assign_fill(cols, 0.0f);
                if (!check_cuda(cudaMemcpy(host_exec_gene_sum.data(), d_gene_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                                issues, "browse", "cudaMemcpy blocked_ell part gene_sum")
                    || !check_cuda(cudaMemcpy(host_exec_gene_detected.data(), d_gene_detected, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                                   issues, "browse", "cudaMemcpy blocked_ell part gene_detected")
                    || !check_cuda(cudaMemcpy(host_exec_gene_sq_sum.data(), d_gene_sq_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                                   issues, "browse", "cudaMemcpy blocked_ell part gene_sq_sum")) {
                    cudaFree(d_gene_sq_sum);
                    cudaFree(d_gene_detected);
                    cudaFree(d_gene_sum);
                    goto done;
                }
                for (unsigned int exec_col = 0u; exec_col < cols; ++exec_col) {
                    const unsigned int canonical_col =
                        exec_part.exec_to_canonical_cols != nullptr ? exec_part.exec_to_canonical_cols[exec_col] : exec_col;
                    out->gene_sum[canonical_col] += host_exec_gene_sum[exec_col];
                    out->gene_detected[canonical_col] += host_exec_gene_detected[exec_col];
                    out->gene_sq_sum[canonical_col] += host_exec_gene_sq_sum[exec_col];
                }
                cellshard::clear(&exec_part);
            }
        }

        cudaFree(d_gene_sq_sum);
        cudaFree(d_gene_detected);
        cudaFree(d_gene_sum);
    }

    out->ok = true;

done:
    cellshard::clear(&exec_part);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return out->ok;
}

[[maybe_unused]] inline bool build_selected_feature_partials(const std::string &path,
                                                             const host_buffer<int> &shard_owner,
                                                             unsigned int worker_slot,
                                                             int device_id,
                                                             const host_buffer<std::uint32_t> &part_dataset_indices,
                                                             unsigned int dataset_count,
                                                             unsigned int shard_count,
                                                             const host_buffer<std::uint32_t> &selected_features,
                                                             unsigned int sample_rows_per_partition,
                                                             host_buffer<float> *dataset_feature_sum,
                                                             host_buffer<float> *shard_feature_sum,
                                                             host_buffer<std::uint64_t> *partition_sample_global_rows,
                                                             host_buffer<float> *partition_sample_values,
                                                             std::vector<issue> *issues) {
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::compressed> device_state;
    unsigned int *d_selected = nullptr;
    unsigned int *d_sample_rows = nullptr;
    float *d_dataset = nullptr;
    float *d_shards = nullptr;
    float *d_tile = nullptr;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !csv::reserve(&device_state, matrix.num_partitions)
        || !check_cuda(cudaSetDevice(device_id), issues, "browse", "cudaSetDevice selected features")) goto done;

    if (!selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_selected, selected_features.size() * sizeof(unsigned int)), issues, "browse", "cudaMalloc selected features")) goto done;
    if (!selected_features.empty()
        && !check_cuda(cudaMemcpy(d_selected,
                                  selected_features.data(),
                                  selected_features.size() * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice),
                       issues, "browse", "cudaMemcpy selected features")) goto done;
    if (sample_rows_per_partition != 0u
        && !check_cuda(cudaMalloc((void **) &d_sample_rows, (std::size_t) sample_rows_per_partition * sizeof(unsigned int)), issues, "browse", "cudaMalloc sample rows")) goto done;
    if (!selected_features.empty()
        && dataset_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_dataset,
                                  (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc dataset feature sums")) goto done;
    if (!selected_features.empty()
        && shard_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_shards,
                                  (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc shard feature sums")) goto done;
    if (d_dataset != nullptr
        && !check_cuda(cudaMemset(d_dataset, 0, (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset dataset feature sums")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemset(d_shards, 0, (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset shard feature sums")) goto done;
    if (sample_rows_per_partition != 0u && !selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_tile,
                                  (std::size_t) sample_rows_per_partition * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc sample tile")) goto done;

    dataset_feature_sum->assign_fill((std::size_t) dataset_count * selected_features.size(), 0.0f);
    shard_feature_sum->assign_fill((std::size_t) shard_count * selected_features.size(), 0.0f);

    for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
        const unsigned long part_begin = cs::first_partition_in_shard(&matrix, shard_id);
        const unsigned long part_end = cs::last_partition_in_shard(&matrix, shard_id);
        for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
            csv::compressed_view part_view;
            const std::size_t tile_offset = (std::size_t) part_id * sample_rows_per_partition * selected_features.size();
            const std::size_t row_offset = (std::size_t) part_id * sample_rows_per_partition;
            const std::uint32_t dataset_id = part_id < part_dataset_indices.size() ? part_dataset_indices[part_id] : 0u;
            const unsigned int blocks = (unsigned int) std::min<unsigned long>(4096ul, (matrix.partition_nnz[part_id] + 255ul) / 256ul);
            const host_buffer<unsigned int> sample_rows = choose_sample_rows((unsigned int) matrix.partition_rows[part_id], sample_rows_per_partition);

            if (!cs::fetch_partition(&matrix, &storage, part_id)
                || !check_cuda(csv::upload_partition(&device_state, &matrix, part_id, device_id), issues, "browse", "upload_partition selected features")
                || !cpre::bind_uploaded_part_view(&part_view, &matrix, device_state.parts + part_id, part_id)) {
                goto done;
            }

            if (!selected_features.empty()) {
                accumulate_selected_feature_sums_kernel<<<std::max(1u, blocks), 256>>>(
                    part_view,
                    d_selected,
                    (unsigned int) selected_features.size(),
                    d_dataset != nullptr ? d_dataset + (std::size_t) dataset_id * selected_features.size() : nullptr,
                    d_shards != nullptr ? d_shards + (std::size_t) shard_id * selected_features.size() : nullptr
                );
                if (!check_cuda(cudaGetLastError(), issues, "browse", "accumulate_selected_feature_sums_kernel")) goto done;
            }

            if (sample_rows_per_partition != 0u && !selected_features.empty()) {
                if (!check_cuda(cudaMemcpy(d_sample_rows,
                                           sample_rows.data(),
                                           (std::size_t) sample_rows_per_partition * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice),
                                issues, "browse", "cudaMemcpy sample rows")) goto done;
                if (!check_cuda(cudaMemset(d_tile,
                                           0,
                                           (std::size_t) sample_rows_per_partition * selected_features.size() * sizeof(float)),
                                issues, "browse", "cudaMemset sample tile")) goto done;
                extract_sample_tile_kernel<<<sample_rows_per_partition, 128>>>(
                    part_view,
                    d_sample_rows,
                    sample_rows_per_partition,
                    d_selected,
                    (unsigned int) selected_features.size(),
                    d_tile
                );
                if (!check_cuda(cudaGetLastError(), issues, "browse", "extract_sample_tile_kernel")) goto done;
                if (!check_cuda(cudaMemcpy(partition_sample_values->data() + tile_offset,
                                           d_tile,
                                           (std::size_t) sample_rows_per_partition * selected_features.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                issues, "browse", "cudaMemcpy sample tile")) goto done;
            }

            if (!check_cuda(cudaDeviceSynchronize(), issues, "browse", "cudaDeviceSynchronize selected features")
                || !check_cuda(csv::release_partition(&device_state, part_id), issues, "browse", "release_partition selected features")) goto done;
            for (unsigned int row = 0; row < sample_rows_per_partition; ++row) {
                const unsigned int local_row = sample_rows[row];
                (*partition_sample_global_rows)[row_offset + row] =
                    local_row < matrix.partition_rows[part_id]
                        ? (std::uint64_t) (matrix.partition_offsets[part_id] + local_row)
                        : std::numeric_limits<std::uint64_t>::max();
            }
            cs::drop_partition(&matrix, part_id);
        }
    }

    if (d_dataset != nullptr
        && !check_cuda(cudaMemcpy(dataset_feature_sum->data(),
                                  d_dataset,
                                  dataset_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy dataset_feature_sum")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemcpy(shard_feature_sum->data(),
                                  d_shards,
                                  shard_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy shard_feature_sum")) goto done;

    ok = true;

done:
    if (d_tile != nullptr) cudaFree(d_tile);
    if (d_shards != nullptr) cudaFree(d_shards);
    if (d_dataset != nullptr) cudaFree(d_dataset);
    if (d_sample_rows != nullptr) cudaFree(d_sample_rows);
    if (d_selected != nullptr) cudaFree(d_selected);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}

inline bool build_selected_feature_partials_blocked_ell(const std::string &path,
                                                        const std::string &cache_dir,
                                                        const host_buffer<int> &shard_owner,
                                                        unsigned int worker_slot,
                                                        int device_id,
                                                        const host_buffer<std::uint32_t> &part_dataset_indices,
                                                        unsigned int dataset_count,
                                                        unsigned int shard_count,
                                                        const host_buffer<std::uint32_t> &selected_features,
                                                        unsigned int sample_rows_per_partition,
                                                        host_buffer<float> *dataset_feature_sum,
                                                        host_buffer<float> *shard_feature_sum,
                                                        host_buffer<std::uint64_t> *partition_sample_global_rows,
                                                        host_buffer<float> *partition_sample_values,
                                                        std::vector<issue> *issues) {
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::blocked_ell> device_state;
    cellshard::bucketed_blocked_ell_partition exec_part;
    unsigned int *d_selected = nullptr;
    unsigned int *d_sample_rows = nullptr;
    float *d_dataset = nullptr;
    float *d_shards = nullptr;
    float *d_tile = nullptr;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);
    cellshard::init(&exec_part);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !cs::bind_dataset_h5_cache(&storage, cache_dir.c_str())
        || !csv::reserve(&device_state, 1ul)
        || !check_cuda(cudaSetDevice(device_id), issues, "browse", "cudaSetDevice blocked_ell selected features")) goto done;

    if (!selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_selected, selected_features.size() * sizeof(unsigned int)),
                       issues, "browse", "cudaMalloc blocked_ell selected features")) goto done;
    if (!selected_features.empty()
        && !check_cuda(cudaMemcpy(d_selected,
                                  selected_features.data(),
                                  selected_features.size() * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice),
                       issues, "browse", "cudaMemcpy blocked_ell selected features")) goto done;
    if (sample_rows_per_partition != 0u
        && !check_cuda(cudaMalloc((void **) &d_sample_rows, (std::size_t) sample_rows_per_partition * sizeof(unsigned int)),
                       issues, "browse", "cudaMalloc blocked_ell sample rows")) goto done;
    if (!selected_features.empty()
        && dataset_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_dataset,
                                  (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc blocked_ell dataset sums")) goto done;
    if (!selected_features.empty()
        && shard_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_shards,
                                  (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc blocked_ell shard sums")) goto done;
    if (d_dataset != nullptr
        && !check_cuda(cudaMemset(d_dataset, 0, (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset blocked_ell dataset sums")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemset(d_shards, 0, (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset blocked_ell shard sums")) goto done;
    if (sample_rows_per_partition != 0u && !selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_tile,
                                  (std::size_t) sample_rows_per_partition * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc blocked_ell sample tile")) goto done;

    dataset_feature_sum->assign_fill((std::size_t) dataset_count * selected_features.size(), 0.0f);
    shard_feature_sum->assign_fill((std::size_t) shard_count * selected_features.size(), 0.0f);

    for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
        const unsigned long part_begin = cs::first_partition_in_shard(&matrix, shard_id);
        const unsigned long part_end = cs::last_partition_in_shard(&matrix, shard_id);
        for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
            const std::size_t tile_offset = (std::size_t) part_id * sample_rows_per_partition * selected_features.size();
            const std::size_t row_offset = (std::size_t) part_id * sample_rows_per_partition;
            const std::uint32_t dataset_id = part_id < part_dataset_indices.size() ? part_dataset_indices[part_id] : 0u;
            const host_buffer<unsigned int> sample_rows = choose_sample_rows((unsigned int) matrix.partition_rows[part_id], sample_rows_per_partition);
            host_buffer<unsigned int> remapped_selected;

            if (!fetch_execution_partition(&exec_part, &matrix, &storage, part_id, issues, "browse", "blocked_ell selected features")) goto done;
            remapped_selected.assign_fill(selected_features.size(), 0u);
            for (std::size_t feature = 0; feature < selected_features.size(); ++feature) {
                const unsigned int canonical_col = selected_features[feature];
                remapped_selected[feature] =
                    exec_part.canonical_to_exec_cols != nullptr ? exec_part.canonical_to_exec_cols[canonical_col] : canonical_col;
            }
            if (!selected_features.empty()
                && !check_cuda(cudaMemcpy(d_selected,
                                          remapped_selected.data(),
                                          remapped_selected.size() * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice),
                               issues, "browse", "cudaMemcpy blocked_ell remapped selected features")) {
                goto done;
            }

            if (!selected_features.empty()) {
                for (std::uint32_t segment = 0u; segment < exec_part.segment_count; ++segment) {
                    csv::blocked_ell_view part_view;
                    single_part_blocked_ell_host host(exec_part.segments + segment);
                    const unsigned long total = (unsigned long) exec_part.segments[segment].rows * (unsigned long) exec_part.segments[segment].ell_cols;
                    const unsigned int blocks = (unsigned int) std::max<unsigned long>(1ul, std::min<unsigned long>(4096ul, (total + 255ul) / 256ul));
                    if (!check_cuda(csv::upload_partition(&device_state, &host.view, 0ul, device_id),
                                    issues, "browse", "upload_execution_segment blocked_ell selected features")
                        || !bind_uploaded_part_view(&part_view, &host.view, device_state.parts, 0ul)) {
                        goto done;
                    }

                    accumulate_selected_feature_sums_blocked_ell_kernel<<<blocks, 256>>>(
                        part_view,
                        d_selected,
                        (unsigned int) selected_features.size(),
                        d_dataset != nullptr ? d_dataset + (std::size_t) dataset_id * selected_features.size() : nullptr,
                        d_shards != nullptr ? d_shards + (std::size_t) shard_id * selected_features.size() : nullptr
                    );
                    if (!check_cuda(cudaGetLastError(), issues, "browse", "accumulate_selected_feature_sums_blocked_ell_kernel")) goto done;

                    if (sample_rows_per_partition != 0u) {
                        host_buffer<unsigned int> segment_sample_rows;
                        host_buffer<float> host_tile;
                        const std::uint32_t seg_row_begin = exec_part.segment_row_offsets[segment];
                        const std::uint32_t seg_row_end = exec_part.segment_row_offsets[segment + 1u];
                        host_tile.assign_fill((std::size_t) sample_rows_per_partition * selected_features.size(), 0.0f);
                        segment_sample_rows.assign_fill(sample_rows_per_partition, std::numeric_limits<unsigned int>::max());
                        for (unsigned int row = 0; row < sample_rows_per_partition; ++row) {
                            if (sample_rows[row] >= exec_part.rows) continue;
                            const std::uint32_t exec_row = exec_part.canonical_to_exec_rows[sample_rows[row]];
                            if (exec_row < seg_row_begin || exec_row >= seg_row_end) continue;
                            segment_sample_rows[row] = exec_row - seg_row_begin;
                        }
                        if (!check_cuda(cudaMemcpy(d_sample_rows,
                                                   segment_sample_rows.data(),
                                                   (std::size_t) sample_rows_per_partition * sizeof(unsigned int),
                                                   cudaMemcpyHostToDevice),
                                        issues, "browse", "cudaMemcpy blocked_ell sample rows")) goto done;
                        if (!check_cuda(cudaMemset(d_tile,
                                                   0,
                                                   (std::size_t) sample_rows_per_partition * selected_features.size() * sizeof(float)),
                                        issues, "browse", "cudaMemset blocked_ell sample tile")) goto done;
                        extract_sample_tile_blocked_ell_kernel<<<sample_rows_per_partition, 128>>>(
                            part_view,
                            d_sample_rows,
                            sample_rows_per_partition,
                            d_selected,
                            (unsigned int) selected_features.size(),
                            d_tile
                        );
                        if (!check_cuda(cudaGetLastError(), issues, "browse", "extract_sample_tile_blocked_ell_kernel")) goto done;
                        if (!check_cuda(cudaMemcpy(host_tile.data(),
                                                   d_tile,
                                                   host_tile.size() * sizeof(float),
                                                   cudaMemcpyDeviceToHost),
                                        issues, "browse", "cudaMemcpy blocked_ell sample tile")) goto done;
                        for (unsigned int row = 0; row < sample_rows_per_partition; ++row) {
                            if (segment_sample_rows[row] == std::numeric_limits<unsigned int>::max()) continue;
                            for (std::size_t feature = 0; feature < selected_features.size(); ++feature) {
                                (*partition_sample_values)[tile_offset + row * selected_features.size() + feature] +=
                                    host_tile[row * selected_features.size() + feature];
                            }
                        }
                    }

                    if (!check_cuda(cudaDeviceSynchronize(), issues, "browse", "cudaDeviceSynchronize blocked_ell selected features")
                        || !check_cuda(csv::release_partition(&device_state, 0ul),
                                       issues, "browse", "release_execution_segment blocked_ell selected features")) goto done;
                }
            }
            for (unsigned int row = 0; row < sample_rows_per_partition; ++row) {
                const unsigned int local_row = sample_rows[row];
                (*partition_sample_global_rows)[row_offset + row] =
                    local_row < matrix.partition_rows[part_id]
                        ? (std::uint64_t) (matrix.partition_offsets[part_id] + local_row)
                        : std::numeric_limits<std::uint64_t>::max();
            }
            cellshard::clear(&exec_part);
        }
    }

    if (d_dataset != nullptr
        && !check_cuda(cudaMemcpy(dataset_feature_sum->data(),
                                  d_dataset,
                                  dataset_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy blocked_ell dataset sums")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemcpy(shard_feature_sum->data(),
                                  d_shards,
                                  shard_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy blocked_ell shard sums")) goto done;

    ok = true;

done:
    cellshard::clear(&exec_part);
    if (d_tile != nullptr) cudaFree(d_tile);
    if (d_shards != nullptr) cudaFree(d_shards);
    if (d_dataset != nullptr) cudaFree(d_dataset);
    if (d_sample_rows != nullptr) cudaFree(d_sample_rows);
    if (d_selected != nullptr) cudaFree(d_selected);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}

inline bool build_browse_cache_multigpu(const std::string &path,
                                        const ingest_plan &plan,
                                        std::vector<issue> *issues) {
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    csd::local_context ctx;
    csd::shard_map shard_map;
    browse_cache_owned owned;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csd::init(&ctx);
    csd::init(&shard_map);

    if (plan.policy.cache_dir.empty()) {
        push_issue(issues, issue_severity::error, "browse", "explicit local cache_dir is required for browse cache generation");
        goto done;
    }

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(issues, issue_severity::error, "browse", "failed to reload dataset header for browse cache build");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, plan.policy.cache_dir.c_str())) {
        push_issue(issues, issue_severity::error, "browse", "failed to bind explicit local cache_dir for browse cache generation");
        goto done;
    }

    if (!check_cuda(csd::discover_local(&ctx, 1, cudaStreamNonBlocking), issues, "browse", "discover_local")
        || !check_cuda(csd::enable_peer_access(&ctx), issues, "browse", "enable_peer_access")) goto done;
    if (ctx.device_count < 4u) {
        push_issue(issues, issue_severity::error, "browse", "browse cache generation requires 4 visible GPUs");
        goto done;
    }
    if (!csd::assign_shards_by_bytes(&shard_map, &matrix, &ctx)) {
        push_issue(issues, issue_severity::error, "browse", "failed to assign shards across GPUs");
        goto done;
    }

    {
        host_buffer<int> shard_owner;
        host_buffer<gene_metric_partial> partials;
        host_buffer<std::thread> workers;
        shard_owner.assign_fill(matrix.num_shards, -1);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                (void) build_gene_metric_partials_blocked_ell(path,
                                                              plan.policy.cache_dir,
                                                              shard_owner,
                                                              slot,
                                                              ctx.device_ids[slot],
                                                              (unsigned int) matrix.cols,
                                                              partials.data() + slot,
                                                              issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        owned.gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        for (const gene_metric_partial &partial : partials) {
            if (!partial.ok) goto done;
            for (std::size_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
                owned.gene_sum[gene] += partial.gene_sum[gene];
                owned.gene_detected[gene] += partial.gene_detected[gene];
                owned.gene_sq_sum[gene] += partial.gene_sq_sum[gene];
            }
        }
    }

    {
        host_buffer<std::pair<float, std::uint32_t>> ranked;
        ranked.reserve(owned.gene_sum.size());
        for (std::uint32_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
            ranked.push_back(std::pair<float, std::uint32_t>{ owned.gene_sum[gene], gene });
        }
        std::partial_sort(ranked.begin(),
                          ranked.begin() + std::min<std::size_t>(plan.policy.browse_top_features, ranked.size()),
                          ranked.end(),
                          [](const auto &lhs, const auto &rhs) {
                              if (lhs.first != rhs.first) return lhs.first > rhs.first;
                              return lhs.second < rhs.second;
                          });
        const std::size_t count = std::min<std::size_t>(plan.policy.browse_top_features, ranked.size());
        owned.selected_feature_indices.assign_fill(count, 0u);
        for (std::size_t i = 0; i < count; ++i) owned.selected_feature_indices[i] = ranked[i].second;
    }

    if (owned.selected_feature_indices.empty()) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache skipped because no features were selected");
        ok = true;
        goto done;
    }

    owned.dataset_feature_mean.assign_fill(plan.datasets.size() * owned.selected_feature_indices.size(), 0.0f);
    owned.shard_feature_mean.assign_fill((std::size_t) matrix.num_shards * owned.selected_feature_indices.size(), 0.0f);
    owned.partition_sample_row_offsets.assign_fill((std::size_t) matrix.num_partitions + 1u, 0u);
    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        owned.partition_sample_row_offsets[part_id + 1u] =
            owned.partition_sample_row_offsets[part_id] + plan.policy.browse_sample_rows_per_partition;
    }
    owned.partition_sample_global_rows.assign_fill((std::size_t) matrix.num_partitions * plan.policy.browse_sample_rows_per_partition,
                                              std::numeric_limits<std::uint64_t>::max());
    owned.partition_sample_values.assign_fill((std::size_t) matrix.num_partitions
                                             * plan.policy.browse_sample_rows_per_partition
                                             * owned.selected_feature_indices.size(),
                                         0.0f);

    {
        host_buffer<int> shard_owner;
        host_buffer<std::uint32_t> part_dataset_indices;
        host_buffer<selected_feature_partial> partials;
        host_buffer<std::thread> workers;
        host_buffer<std::uint32_t> source_to_dataset;
        shard_owner.assign_fill(matrix.num_shards, -1);
        part_dataset_indices.assign_fill(plan.parts.size(), 0u);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        source_to_dataset.assign_fill(plan.sources.size(), std::numeric_limits<std::uint32_t>::max());
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }
        for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
            source_to_dataset[plan.datasets[dataset_index].source_index] = (std::uint32_t) dataset_index;
        }
        for (std::size_t part_index = 0; part_index < plan.parts.size(); ++part_index) {
            const std::size_t source_index = plan.parts[part_index].source_index;
            if (source_index < source_to_dataset.size() && source_to_dataset[source_index] != std::numeric_limits<std::uint32_t>::max()) {
                part_dataset_indices[part_index] = source_to_dataset[source_index];
            }
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                partials[slot].ok = build_selected_feature_partials_blocked_ell(path,
                                                                                plan.policy.cache_dir,
                                                                                shard_owner,
                                                                                slot,
                                                                                ctx.device_ids[slot],
                                                                                part_dataset_indices,
                                                                                (unsigned int) plan.datasets.size(),
                                                                                (unsigned int) matrix.num_shards,
                                                                                owned.selected_feature_indices,
                                                                                plan.policy.browse_sample_rows_per_partition,
                                                                                &partials[slot].dataset_feature_sum,
                                                                                &partials[slot].shard_feature_sum,
                                                                                &owned.partition_sample_global_rows,
                                                                                &owned.partition_sample_values,
                                                                                issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        for (const selected_feature_partial &partial : partials) {
            if (!partial.ok) goto done;
            if (partial.dataset_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.dataset_feature_mean.size(); ++i) {
                owned.dataset_feature_mean[i] += partial.dataset_feature_sum[i];
            }
        }
        for (const selected_feature_partial &partial : partials) {
            if (partial.shard_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.shard_feature_mean.size(); ++i) {
                owned.shard_feature_mean[i] += partial.shard_feature_sum[i];
            }
        }
    }

    for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
        const double denom = plan.datasets[dataset_index].rows != 0 ? (double) plan.datasets[dataset_index].rows : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] =
                (float) (owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] / denom);
        }
    }
    for (std::size_t shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        const double denom = cs::rows_in_shard(&matrix, (unsigned long) shard_id) != 0
            ? (double) cs::rows_in_shard(&matrix, (unsigned long) shard_id)
            : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] =
                (float) (owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] / denom);
        }
    }

    {
        cellshard::dataset_browse_cache_view view{};
        view.selected_feature_count = (std::uint32_t) owned.selected_feature_indices.size();
        view.selected_feature_indices = owned.selected_feature_indices.data();
        view.gene_sum = owned.gene_sum.data();
        view.gene_detected = owned.gene_detected.data();
        view.gene_sq_sum = owned.gene_sq_sum.data();
        view.dataset_count = (std::uint32_t) plan.datasets.size();
        view.dataset_feature_mean = owned.dataset_feature_mean.data();
        view.shard_count = (std::uint32_t) matrix.num_shards;
        view.shard_feature_mean = owned.shard_feature_mean.data();
        view.partition_count = (std::uint32_t) matrix.num_partitions;
        view.sample_rows_per_partition = plan.policy.browse_sample_rows_per_partition;
        view.partition_sample_row_offsets = owned.partition_sample_row_offsets.data();
        view.partition_sample_global_rows = owned.partition_sample_global_rows.data();
        view.partition_sample_values = owned.partition_sample_values.data();
        if (!cellshard::append_dataset_browse_cache_h5(path.c_str(), &view)) {
            push_issue(issues, issue_severity::error, "browse", "failed to append browse cache to dataset.csh5");
            goto done;
        }
    }

    ok = true;

done:
    csd::clear(&shard_map);
    csd::clear(&ctx);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}

} // namespace

conversion_report convert_plan_to_dataset_csh5(const ingest_plan &plan) {
    conversion_report report;
    cseries::manifest manifest;
    cseries::dataset_h5_convert_options options;
    cs::sharded<cs::sparse::blocked_ell> header;
    cs::shard_storage storage;
    cseries::init(&manifest);
    cseries::init(&options);
    cs::init(&header);
    cs::init(&storage);

    report.events.push_back({"validate", "validating ingest plan"});
    if (!plan.ok) {
        push_issue(&report.issues, issue_severity::error, "convert", "ingest plan is not valid");
        return report;
    }
    if (plan.policy.output_path.empty()) {
        push_issue(&report.issues, issue_severity::error, "convert", "output path is empty");
        return report;
    }

    for (const source_entry &source : plan.sources) {
        if (!source.included) continue;
        if (!cseries::append(&manifest,
                             source.dataset_id.c_str(),
                             source.matrix_path.c_str(),
                             source.format,
                             source.feature_path.c_str(),
                             source.barcode_path.c_str(),
                             source.metadata_path.c_str(),
                             source.matrix_source.c_str(),
                             source.allow_processed ? 1u : 0u,
                             source.rows,
                             source.cols,
                             source.nnz)) {
            push_issue(&report.issues, issue_severity::error, "convert", "failed to materialize manifest from ingest plan");
            cseries::clear(&manifest);
            return report;
        }
    }

    options.max_part_nnz = plan.policy.max_part_nnz;
    options.convert_window_bytes = plan.policy.convert_window_bytes;
    options.target_shard_bytes = plan.policy.target_shard_bytes;
    options.reader_bytes = plan.policy.reader_bytes;
    options.cache_root = plan.policy.cache_dir;
    options.device = plan.policy.device;
    options.stream = (cudaStream_t) 0;

    report.events.push_back({"convert", "writing dataset.csh5"});
    if (!cseries::convert_manifest_dataset_to_hdf5(&manifest, plan.policy.output_path.c_str(), &options)) {
        push_issue(&report.issues, issue_severity::error, "convert", "dataset ingest conversion failed");
        cseries::clear(&manifest);
        return report;
    }

    if (plan.policy.embed_metadata) {
        report.events.push_back({"metadata", "embedding metadata tables"});
        if (!append_embedded_metadata_tables(plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
        report.events.push_back({"metadata", "embedding typed observation metadata"});
        if (!append_observation_metadata_table(plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
    }

    report.events.push_back({"execution", "writing execution layout metadata"});
    if (!append_execution_layout_metadata(plan, &report.issues)) {
        cseries::clear(&manifest);
        return report;
    }

    if (!plan.policy.cache_dir.empty()) {
        report.events.push_back({"execution", "warming bucketed execution cache"});
        if (!cs::warm_dataset_blocked_ell_h5_execution_cache(plan.policy.output_path.c_str(), plan.policy.cache_dir.c_str())) {
            push_issue(&report.issues, issue_severity::error, "execution", "failed to warm bucketed execution cache");
            cseries::clear(&manifest);
            return report;
        }
    }

    if (plan.policy.build_browse_cache) {
        report.events.push_back({"browse", "building 4-GPU browse cache"});
        if (!build_browse_cache_multigpu(plan.policy.output_path, plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
    }

    if (plan.policy.verify_after_write) {
        report.events.push_back({"verify", "loading written header"});
        if (!cs::load_header(plan.policy.output_path.c_str(), &header, &storage)) {
            push_issue(&report.issues, issue_severity::error, "verify", "failed to reopen written dataset.csh5 header");
            cseries::clear(&manifest);
            cs::clear(&storage);
            cs::clear(&header);
            return report;
        }
        if (header.rows != plan.total_rows || header.num_partitions != plan.parts.size() || header.num_shards != plan.shards.size()) {
            push_issue(&report.issues, issue_severity::error, "verify", "written header does not match the planned layout");
            cseries::clear(&manifest);
            cs::clear(&storage);
            cs::clear(&header);
            return report;
        }
    }

    report.events.push_back({"done", "conversion completed"});
    report.ok = true;
    cseries::clear(&manifest);
    cs::clear(&storage);
    cs::clear(&header);
    return report;
}

preprocess_summary run_preprocess_pass(const std::string &path, const preprocess_config &config) {
    preprocess_summary summary;
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    cellshard::bucketed_blocked_ell_partition exec_part;
    csv::partition_record<cs::sparse::sliced_ell> device_part;
    cpre::device_workspace workspace;
    dataset_summary dataset = summarize_dataset_csh5(path);
    host_buffer<unsigned char> gene_flags;
    host_buffer<unsigned char> host_keep_genes;
    host_buffer<float> host_gene_sum;
    host_buffer<float> host_gene_sq_sum;
    host_buffer<float> host_gene_detected;
    host_buffer<float> host_cell_total_counts;
    host_buffer<float> host_cell_mito_counts;
    host_buffer<float> host_cell_max_counts;
    host_buffer<unsigned int> host_cell_detected_genes;
    host_buffer<unsigned char> host_keep_cells;
    float kept_cells = 0.0f;
    unsigned int mito_feature_count = 0u;
    const cpre::cell_filter_params cell_filter = {
        config.min_counts,
        config.min_genes,
        config.max_mito_fraction
    };
    const cpre::gene_filter_params gene_filter = {
        config.min_gene_sum,
        config.min_detected_cells,
        config.min_variance
    };

    cs::init(&matrix);
    cs::init(&storage);
    cellshard::init(&exec_part);
    csv::zero_record(&device_part);
    cpre::init(&workspace);

    if (!dataset.ok) {
        summary.issues = dataset.issues;
        push_issue(&summary.issues, issue_severity::error, "preprocess", "cannot preprocess an unreadable dataset.csh5");
        return summary;
    }
    if (dataset.preprocess.available) {
        push_issue(&summary.issues,
                   issue_severity::error,
                   "preprocess",
                   "dataset.csh5 already contains persisted preprocess metadata; start from a raw dataset file instead");
        return summary;
    }
    if (config.cache_dir.empty()) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "explicit local cache_dir is required for Blocked-ELL preprocess");
        return summary;
    }

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), &summary.issues, "preprocess", "cudaGetDeviceCount")) return summary;
    if (device_count == 0) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "no CUDA devices are available");
        return summary;
    }
    if (config.device < 0 || config.device >= device_count) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "requested CUDA device is out of range");
        return summary;
    }

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to load dataset header");
        return summary;
    }
    if (!cs::bind_dataset_h5_cache(&storage, config.cache_dir.c_str())) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to bind explicit local cache_dir");
        return summary;
    }

    if (!cpre::setup(&workspace, config.device, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_partition_rows(matrix), (unsigned int) matrix.cols, max_partition_nnz(matrix))) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to set up preprocess workspace");
        goto done;
    }

    host_cell_total_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_mito_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_max_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_detected_genes.assign_fill((std::size_t) matrix.rows, 0u);
    host_keep_cells.assign_fill((std::size_t) matrix.rows, static_cast<unsigned char>(0u));
    host_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
    gene_flags = build_gene_flags(dataset, config);

    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        host_buffer<unsigned char> exec_gene_flags;
        host_buffer<float> exec_gene_sum;
        host_buffer<float> exec_gene_sq_sum;
        host_buffer<float> exec_gene_detected;
        if (!fetch_execution_partition(&exec_part, &matrix, &storage, part_id, &summary.issues, "preprocess", "preprocess")) goto done;
        exec_gene_flags.assign_fill((std::size_t) matrix.cols, static_cast<unsigned char>(0u));
        for (unsigned int exec_col = 0u; exec_col < (unsigned int) matrix.cols; ++exec_col) {
            const unsigned int canonical_col =
                exec_part.exec_to_canonical_cols != nullptr ? exec_part.exec_to_canonical_cols[exec_col] : exec_col;
            exec_gene_flags[exec_col] = canonical_col < gene_flags.size() ? gene_flags[canonical_col] : static_cast<unsigned char>(0u);
        }
        if (!cpre::upload_gene_flags(&workspace, (unsigned int) matrix.cols, exec_gene_flags.empty() ? nullptr : exec_gene_flags.data())
            || !cpre::zero_gene_metrics(&workspace, (unsigned int) matrix.cols)) {
            push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to initialize preprocess workspace state");
            goto done;
        }
        for (std::uint32_t segment = 0u; segment < exec_part.segment_count; ++segment) {
            csv::sliced_ell_view part_view;
            cpre::part_preprocess_result part_result;
            owned_sliced_ell_host host;
            host_buffer<float> seg_total_counts;
            host_buffer<float> seg_mito_counts;
            host_buffer<float> seg_max_counts;
            host_buffer<unsigned int> seg_detected_genes;
            host_buffer<unsigned char> seg_keep_cells;
            if (!build_preprocess_sliced_segment(exec_part.segments + segment, 32u, &host.part)) {
                push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to build sliced preprocess segment");
                goto done;
            }
            if (!check_cuda(cudaSetDevice(config.device), &summary.issues, "preprocess", "cudaSetDevice upload sliced preprocess segment")
                || !check_cuda(csv::upload(&host.part, &device_part), &summary.issues, "preprocess", "upload_sliced_execution_segment")) {
                goto done;
            }
            if (!cpre::bind_uploaded_part_view(&part_view, &host.part, &device_part)) {
                push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to bind uploaded sliced execution segment");
                goto done;
            }
            if (!cpre::preprocess_part_inplace(&part_view, &workspace, cell_filter, config.target_sum, &part_result)) {
                push_issue(&summary.issues, issue_severity::error, "preprocess", "GPU sliced preprocess kernel pass failed");
                goto done;
            }
            if (!check_cuda(cudaStreamSynchronize(workspace.stream), &summary.issues, "preprocess", "cudaStreamSynchronize part")) goto done;
            if (part_result.cell.rows != 0u) {
                const std::size_t row_count = (std::size_t) part_result.cell.rows;
                seg_total_counts.assign_fill(row_count, 0.0f);
                seg_mito_counts.assign_fill(row_count, 0.0f);
                seg_max_counts.assign_fill(row_count, 0.0f);
                seg_detected_genes.assign_fill(row_count, 0u);
                seg_keep_cells.assign_fill(row_count, static_cast<unsigned char>(0u));
                if (!check_cuda(cudaMemcpy(seg_total_counts.data(),
                                           part_result.cell.total_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &summary.issues, "preprocess", "cudaMemcpy cell total counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_mito_counts.data(),
                                           part_result.cell.mito_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &summary.issues, "preprocess", "cudaMemcpy cell mito counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_max_counts.data(),
                                           part_result.cell.max_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &summary.issues, "preprocess", "cudaMemcpy cell max counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_detected_genes.data(),
                                           part_result.cell.detected_genes,
                                           row_count * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost),
                                &summary.issues, "preprocess", "cudaMemcpy cell detected genes")) goto done;
                if (!check_cuda(cudaMemcpy(seg_keep_cells.data(),
                                           part_result.cell.keep_cells,
                                           row_count * sizeof(unsigned char),
                                           cudaMemcpyDeviceToHost),
                                &summary.issues, "preprocess", "cudaMemcpy cell keep mask")) goto done;
                for (std::size_t local_row = 0; local_row < row_count; ++local_row) {
                    const std::size_t exec_row = (std::size_t) exec_part.segment_row_offsets[segment] + local_row;
                    const std::size_t canonical_row = (std::size_t) exec_part.exec_to_canonical_rows[exec_row];
                    const std::size_t dst_row = (std::size_t) matrix.partition_offsets[part_id] + canonical_row;
                    host_cell_total_counts[dst_row] = seg_total_counts[local_row];
                    host_cell_mito_counts[dst_row] = seg_mito_counts[local_row];
                    host_cell_max_counts[dst_row] = seg_max_counts[local_row];
                    host_cell_detected_genes[dst_row] = seg_detected_genes[local_row];
                    host_keep_cells[dst_row] = seg_keep_cells[local_row];
                }
            }
            if (!check_cuda(csv::release(&device_part), &summary.issues, "preprocess", "release_sliced_execution_segment")) goto done;
        }
        exec_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        if (!check_cuda(cudaMemcpy(exec_gene_sum.data(),
                                   workspace.d_gene_sum,
                                   exec_gene_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &summary.issues,
                        "preprocess",
                        "cudaMemcpy part gene sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_sq_sum.data(),
                                   workspace.d_gene_sq_sum,
                                   exec_gene_sq_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &summary.issues,
                        "preprocess",
                        "cudaMemcpy part gene sq sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_detected.data(),
                                   workspace.d_gene_detected,
                                   exec_gene_detected.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &summary.issues,
                        "preprocess",
                        "cudaMemcpy part gene detected")) goto done;
        for (unsigned int exec_col = 0u; exec_col < (unsigned int) matrix.cols; ++exec_col) {
            const unsigned int canonical_col =
                exec_part.exec_to_canonical_cols != nullptr ? exec_part.exec_to_canonical_cols[exec_col] : exec_col;
            host_gene_sum[canonical_col] += exec_gene_sum[exec_col];
            host_gene_sq_sum[canonical_col] += exec_gene_sq_sum[exec_col];
            host_gene_detected[canonical_col] += exec_gene_detected[exec_col];
        }
        ++summary.partitions_processed;
        cellshard::clear(&exec_part);
    }
    host_keep_genes.assign_fill((std::size_t) matrix.cols, static_cast<unsigned char>(0u));
    kept_cells = 0.0f;
    for (unsigned char keep : host_keep_cells) kept_cells += keep != 0u ? 1.0f : 0.0f;
    {
        const float inv_cells = kept_cells > 0.0f ? (1.0f / kept_cells) : 0.0f;
        for (unsigned int gene = 0u; gene < (unsigned int) matrix.cols; ++gene) {
            const float mean = host_gene_sum[gene] * inv_cells;
            const float var = std::max(host_gene_sq_sum[gene] * inv_cells - mean * mean, 0.0f);
            host_keep_genes[gene] =
                (unsigned char) (host_gene_sum[gene] >= gene_filter.min_sum
                                 && host_gene_detected[gene] >= gene_filter.min_detected_cells
                                 && var >= gene_filter.min_variance);
        }
    }

    summary.device = config.device;
    summary.rows = matrix.rows;
    summary.cols = matrix.cols;
    summary.nnz = matrix.nnz;
    summary.kept_cells = kept_cells;
    for (unsigned char flag : gene_flags) mito_feature_count += (flag & cpre::gene_flag_mito) != 0u;
    for (unsigned char keep : host_keep_genes) summary.kept_genes += keep != 0;
    for (float value : host_gene_sum) summary.gene_sum_checksum += (double) value;

    // HDF5 rejects reopening the same dataset file read-write while the read-only header
    // backend is still alive. Persist preprocess metadata only after releasing that handle.
    cs::clear(&storage);
    cs::clear(&matrix);

    {
        cs::dataset_preprocess_view preprocess = {};
        preprocess.assay = "scrna";
        preprocess.matrix_orientation = "observations_by_features";
        preprocess.matrix_state = "raw_counts";
        preprocess.pipeline_scope = "qc_filter_metrics_from_normalized_log1p";
        preprocess.raw_matrix_name = "X_raw";
        preprocess.active_matrix_name = "X_raw";
        preprocess.feature_namespace = "unknown";
        preprocess.mito_prefix = config.mito_prefix.c_str();
        preprocess.raw_counts_available = 1u;
        preprocess.processed_matrix_available = 0u;
        preprocess.normalized_log1p_metrics = 1u;
        preprocess.hvg_available = 0u;
        preprocess.mark_mito_from_feature_names = config.mark_mito_from_feature_names ? 1u : 0u;
        preprocess.rows = summary.rows;
        preprocess.cols = (std::uint32_t) summary.cols;
        preprocess.nnz = summary.nnz;
        preprocess.partitions_processed = (std::uint32_t) summary.partitions_processed;
        preprocess.mito_feature_count = mito_feature_count;
        preprocess.target_sum = config.target_sum;
        preprocess.min_counts = config.min_counts;
        preprocess.min_genes = config.min_genes;
        preprocess.max_mito_fraction = config.max_mito_fraction;
        preprocess.min_gene_sum = config.min_gene_sum;
        preprocess.min_detected_cells = config.min_detected_cells;
        preprocess.min_variance = config.min_variance;
        preprocess.kept_cells = summary.kept_cells;
        preprocess.kept_genes = (std::uint32_t) summary.kept_genes;
        preprocess.gene_sum_checksum = summary.gene_sum_checksum;
        preprocess.cell_total_counts = host_cell_total_counts.empty() ? nullptr : host_cell_total_counts.data();
        preprocess.cell_mito_counts = host_cell_mito_counts.empty() ? nullptr : host_cell_mito_counts.data();
        preprocess.cell_max_counts = host_cell_max_counts.empty() ? nullptr : host_cell_max_counts.data();
        preprocess.cell_detected_genes = host_cell_detected_genes.empty() ? nullptr : host_cell_detected_genes.data();
        preprocess.cell_keep = host_keep_cells.empty() ? nullptr : host_keep_cells.data();
        preprocess.gene_sum = host_gene_sum.empty() ? nullptr : host_gene_sum.data();
        preprocess.gene_sq_sum = host_gene_sq_sum.empty() ? nullptr : host_gene_sq_sum.data();
        preprocess.gene_detected_cells = host_gene_detected.empty() ? nullptr : host_gene_detected.data();
        preprocess.gene_keep = host_keep_genes.empty() ? nullptr : host_keep_genes.data();
        preprocess.gene_flags = gene_flags.empty() ? nullptr : gene_flags.data();
        if (!cs::append_dataset_preprocess_h5(path.c_str(), &preprocess)) {
            push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to append persisted preprocess metadata");
            goto done;
        }
    }

    summary.ok = true;

done:
    if (device_part.view != nullptr) (void) csv::release(&device_part);
    cellshard::clear(&exec_part);
    cpre::clear(&workspace);
    cs::clear(&storage);
    cs::clear(&matrix);
    return summary;
}

} // namespace cellerator::apps::workbench
