#include "../dataset_workbench.hh"

#include <MosaiCell/runtime.hh>

#include "../../compute/preprocess/operators.cuh"
#include "../../compute/preprocess/workspace.cuh"
#include "../../ingest/common/metadata_table.cuh"
#include "../../ingest/h5ad/h5ad_reader.cuh"
#include "../../ingest/dataset/dataset_ingest.cuh"

#include "../../../extern/CellShard/include/CellShard/runtime/storage/disk.cuh"
#include "../../../extern/CellShard/include/CellShard/runtime/distributed/distributed.cuh"
#include "../../../extern/CellShard/include/CellShard/runtime/device/sharded_device.cuh"
#include "../../../extern/CellShard/include/CellShard/runtime/host/sharded_host.cuh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
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
namespace mosaic = ::mosaicell;

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

#include "internal/orchestration_support_part.hh"

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
                                            std::uint32_t fallback_slice_rows,
                                            int device,
                                            cs::sparse::sliced_ell *out) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u, 64u };
    if (src == nullptr || out == nullptr || fallback_slice_rows == 0u) return false;
    if (cs::convert::sliced_ell_from_blocked_ell_cuda_auto(src,
                                                           candidates,
                                                           sizeof(candidates) / sizeof(candidates[0]),
                                                           out,
                                                           device,
                                                           (cudaStream_t) 0,
                                                           nullptr)) {
        return true;
    }
    return cs::convert::sliced_ell_from_blocked_ell_cuda(src, fallback_slice_rows, out, device, (cudaStream_t) 0) != 0;
}

inline bool fetch_execution_partition(cellshard::bucketed_blocked_ell_partition *out,
                                      cs::sharded<cs::sparse::blocked_ell> *matrix,
                                      cs::shard_storage *storage,
                                      unsigned long part_id,
                                      std::vector<issue> *issues,
                                      const std::string &scope,
                                      const std::string &label) {
    if (!cs::fetch_dataset_blocked_ell_h5_pack_partition(out, matrix, storage, part_id)) {
        push_issue(issues, issue_severity::error, scope, "failed to fetch bucketed pack partition for " + label);
        return false;
    }
    return true;
}

inline bool fetch_execution_partition(cellshard::bucketed_sliced_ell_partition *out,
                                      cs::sharded<cs::sparse::sliced_ell> *matrix,
                                      cs::shard_storage *storage,
                                      unsigned long part_id,
                                      std::vector<issue> *issues,
                                      const std::string &scope,
                                      const std::string &label) {
    if (!cs::fetch_dataset_sliced_ell_h5_bucketed_partition(out, matrix, storage, part_id)) {
        push_issue(issues, issue_severity::error, scope, "failed to fetch bucketed pack partition for " + label);
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

struct staged_annotation_column {
    std::string name;
    std::uint32_t type = cs::dataset_observation_metadata_type_none;
    std::vector<std::string> text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;
};

struct owned_embedded_metadata_bundle {
    std::vector<std::unique_ptr<owned_metadata_table>> tables;
    std::vector<cs::dataset_metadata_table_view> table_views;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> global_row_begin;
    std::vector<std::uint64_t> global_row_end;
    cs::dataset_embedded_metadata_view view() const {
        cs::dataset_embedded_metadata_view out{};
        out.count = (std::uint32_t) table_views.size();
        out.dataset_indices = dataset_indices.empty() ? nullptr : dataset_indices.data();
        out.global_row_begin = global_row_begin.empty() ? nullptr : global_row_begin.data();
        out.global_row_end = global_row_end.empty() ? nullptr : global_row_end.data();
        out.tables = table_views.empty() ? nullptr : table_views.data();
        return out;
    }
};

struct owned_annotation_bundle {
    std::vector<std::unique_ptr<owned_observation_metadata_column>> columns;
    std::vector<cs::dataset_observation_metadata_column_view> views;
};

struct owned_user_attribute_bundle {
    ccommon::text_column keys;
    ccommon::text_column values;

    owned_user_attribute_bundle() {
        ccommon::init(&keys);
        ccommon::init(&values);
    }
    ~owned_user_attribute_bundle() {
        ccommon::clear(&values);
        ccommon::clear(&keys);
    }
    owned_user_attribute_bundle(const owned_user_attribute_bundle &) = delete;
    owned_user_attribute_bundle &operator=(const owned_user_attribute_bundle &) = delete;

    cs::dataset_user_attribute_view view(std::uint32_t count) const {
        cs::dataset_user_attribute_view out{};
        out.count = count;
        out.keys = as_text_view(&keys);
        out.values = as_text_view(&values);
        return out;
    }
};

#include "internal/metadata_support_part.hh"

#include "kernels/accumulate_selected_feature_sums_kernel.cuh"
#include "kernels/extract_sample_tile_kernel.cuh"
#include "kernels/accumulate_gene_metrics_blocked_ell_kernel.cuh"
#include "kernels/accumulate_selected_feature_sums_blocked_ell_kernel.cuh"
#include "kernels/extract_sample_tile_blocked_ell_kernel.cuh"

#include "internal/metadata_io_part.hh"

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

#include "internal/browse_cache_build_part.hh"

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
    options.working_root = plan.policy.working_root;
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
        report.events.push_back({"metadata", "embedding feature metadata"});
        if (!append_feature_metadata_table(plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
    }

    report.events.push_back({"execution", "writing execution metadata"});
    if (!append_execution_layout_metadata(plan, &report.issues)) {
        cseries::clear(&manifest);
        return report;
    }

    if (!plan.policy.cache_dir.empty()) {
        report.events.push_back({"execution", "warming sliced cache"});
        if (!cs::warm_dataset_sliced_ell_h5_cache(plan.policy.output_path.c_str(), plan.policy.cache_dir.c_str())) {
            push_issue(&report.issues, issue_severity::error, "execution", "failed to warm sliced cache");
            cseries::clear(&manifest);
            return report;
        }
    }

    if (plan.policy.build_browse_cache) {
        const dataset_summary converted = summarize_dataset_csh5(plan.policy.output_path);
        if (!converted.ok) {
            push_issue(&report.issues, issue_severity::error, "browse", "failed to summarize converted dataset before browse build");
            cseries::clear(&manifest);
            return report;
        }
        if (converted.matrix_format.find("blocked") != std::string::npos) {
            report.events.push_back({"browse", "building 4-GPU browse cache"});
            if (!build_browse_cache_multigpu(plan.policy.output_path, plan, &report.issues)) {
                cseries::clear(&manifest);
                return report;
            }
        } else {
            report.events.push_back({"browse", "skipping browse build until blocked finalize"});
        }
    }

    if (plan.policy.verify_after_write) {
        report.events.push_back({"verify", "loading written header"});
        const dataset_summary written = summarize_dataset_csh5(plan.policy.output_path);
        if (!written.ok) {
            push_issue(&report.issues, issue_severity::error, "verify", "failed to reopen written dataset.csh5 header");
            cseries::clear(&manifest);
            return report;
        }
        if (written.rows != plan.total_rows
            || written.partitions.size() != plan.parts.size()
            || written.shards.size() != plan.shards.size()) {
            push_issue(&report.issues, issue_severity::error, "verify", "written header does not match the planned layout");
            cseries::clear(&manifest);
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

#include "internal/preprocess_runtime_part.hh"

} // namespace cellerator::apps::workbench
