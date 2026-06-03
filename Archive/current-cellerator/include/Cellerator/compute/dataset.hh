#pragma once

#include <Cellerator/core/matrix/compressed.cuh>

#include <CellShard/export/dataset_export.hh>
#include <CellShard/formats/compressed.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compute::dataset {

namespace cm = ::cellerator::core::matrix;
namespace cse = ::cellshard::exporting;

enum class dataset_matrix_kind : std::uint32_t {
    unknown = 0u,
    cellshard_file = 1u,
    owned_compressed = 2u,
    cellshard_sharded_compressed = 3u
};

enum class dataset_layout_kind : std::uint32_t {
    unknown = 0u,
    compressed_csr = 1u,
    blocked_ell = 2u,
    sliced_ell = 3u
};

struct dataset_summary_view {
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    dataset_layout_kind layout = dataset_layout_kind::unknown;
    const char *matrix_state = nullptr;
};

struct row_label_view {
    const char * const *labels = nullptr;
    std::uint64_t count = 0u;
};

struct row_group_span {
    std::string name;
    std::uint64_t begin = 0u;
    std::uint64_t end = 0u;
};

struct stratified_sample_request {
    std::uint64_t row_count = 0u;
    row_label_view labels;
    std::uint64_t max_rows_per_stratum = 0u;
    std::uint64_t max_total_rows = 0u;
    std::uint64_t seed = 0u;
};

struct stratified_sample_plan {
    std::vector<std::uint64_t> row_indices;
    std::vector<row_group_span> row_groups;
};

struct dataset_matrix_handle {
    dataset_matrix_kind kind = dataset_matrix_kind::unknown;
    dataset_summary_view summary;
    const char *cellshard_path = nullptr;
    const cm::compressed *compressed = nullptr;
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *cellshard_compressed = nullptr;
};

struct stratified_downsample_request {
    row_label_view labels;
    const char *cellshard_label_column = nullptr;
    std::uint64_t max_rows_per_stratum = 0u;
    std::uint64_t max_total_rows = 0u;
    std::uint64_t seed = 0u;
};

struct owned_dataset_artifact {
    cm::compressed matrix;
    dataset_summary_view summary;
    std::vector<std::uint64_t> source_row_indices;
    std::vector<std::string> source_row_labels;
    std::vector<row_group_span> row_groups;

    owned_dataset_artifact();
    ~owned_dataset_artifact();
    owned_dataset_artifact(const owned_dataset_artifact &) = delete;
    owned_dataset_artifact &operator=(const owned_dataset_artifact &) = delete;
    owned_dataset_artifact(owned_dataset_artifact &&other) noexcept;
    owned_dataset_artifact &operator=(owned_dataset_artifact &&other) noexcept;

    void clear();
};

dataset_matrix_handle make_cellshard_file_handle(const char *path);
dataset_matrix_handle make_compressed_handle(const cm::compressed *matrix);
dataset_matrix_handle make_cellshard_sharded_compressed_handle(
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *matrix);

bool build_stratified_row_sample(const stratified_sample_request &request,
                                 stratified_sample_plan *out,
                                 std::string *error = nullptr);

bool load_cellshard_row_labels(const char *path,
                               const char *column_name,
                               std::vector<std::string> *out,
                               std::string *error = nullptr);

bool rebuild_rows_as_compressed(const cm::compressed *source,
                                const std::uint64_t *row_indices,
                                std::uint64_t row_count,
                                cm::compressed *out,
                                std::string *error = nullptr);

bool rebuild_cellshard_rows_as_compressed(const char *path,
                                          const std::uint64_t *row_indices,
                                          std::uint64_t row_count,
                                          cm::compressed *out,
                                          std::string *error = nullptr);

bool rebuild_sharded_compressed_rows_as_compressed(
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *source,
    const std::uint64_t *row_indices,
    std::uint64_t row_count,
    cm::compressed *out,
    std::string *error = nullptr);

bool build_stratified_downsample(const dataset_matrix_handle &source,
                                 const stratified_downsample_request &request,
                                 owned_dataset_artifact *out,
                                 std::string *error = nullptr);

} // namespace cellerator::compute::dataset
