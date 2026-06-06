#include <Cellerator/compute/dataset.hh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cd = cellerator::compute::dataset;
namespace cm = cellerator::core::matrix;
namespace ct = cellerator::core::types;

int require(bool ok, const char *label) {
    if (!ok) std::fprintf(stderr, "%s\n", label);
    return ok ? 1 : 0;
}

bool close_f32(float a, float b) {
    return std::fabs(a - b) < 0.001f;
}

bool fill_source(cm::compressed *m) {
    cm::init(m, 5u, 4u, 9u, cm::compressed_by_row);
    if (!cm::allocate(m)) return false;
    const ct::ptr_t ptr[] = {0u, 2u, 4u, 5u, 7u, 9u};
    const ct::idx_t idx[] = {0u, 2u, 1u, 3u, 2u, 0u, 3u, 1u, 2u};
    const float val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for (std::size_t i = 0; i < 6u; ++i) m->majorPtr[i] = ptr[i];
    for (std::size_t i = 0; i < 9u; ++i) {
        m->minorIdx[i] = idx[i];
        m->val[i] = __float2half(val[i]);
    }
    return true;
}

int test_stratified_plan() {
    const char *labels[] = {"late", "early", "late", "early", "late"};
    cd::stratified_sample_request req;
    cd::stratified_sample_plan a, b;
    std::string error;
    req.row_count = 5u;
    req.labels = {labels, 5u};
    req.max_rows_per_stratum = 2u;
    req.seed = 11u;
    if (!require(cd::build_stratified_row_sample(req, &a, &error), error.c_str())) return 1;
    if (!require(cd::build_stratified_row_sample(req, &b, &error), error.c_str())) return 2;
    if (!require(a.row_indices == b.row_indices, "stratified sample is not deterministic")) return 3;
    if (!require(a.row_groups.size() == 2u, "expected two strata")) return 4;
    if (!require(a.row_indices.size() == 4u, "expected capped row count")) return 5;
    if (!require(a.row_groups[0].name == "early" && a.row_groups[0].begin == 0u && a.row_groups[0].end == 2u,
                 "early group span mismatch")) return 6;
    if (!require(a.row_groups[1].name == "late" && a.row_groups[1].begin == 2u && a.row_groups[1].end == 4u,
                 "late group span mismatch")) return 7;
    return 0;
}

int test_in_memory_rebuild() {
    cm::compressed source;
    cd::owned_dataset_artifact artifact;
    const char *labels[] = {"late", "early", "late", "early", "late"};
    cd::stratified_downsample_request req;
    std::string error;
    cm::init(&source);
    if (!require(fill_source(&source), "failed to allocate source matrix")) return 10;
    req.labels = {labels, 5u};
    req.max_rows_per_stratum = 2u;
    req.seed = 11u;
    const cd::dataset_matrix_handle handle = cd::make_compressed_handle(&source);
    if (!require(cd::build_stratified_downsample(handle, req, &artifact, &error), error.c_str())) {
        cm::clear(&source);
        return 11;
    }
    if (!require(artifact.matrix.rows == 4u && artifact.matrix.cols == 4u, "rebuilt shape mismatch")) return 12;
    if (!require(artifact.matrix.axis == cm::compressed_by_row, "rebuilt matrix is not row compressed")) return 13;
    if (!require(artifact.source_row_indices.size() == 4u, "source row provenance missing")) return 14;
    if (!require(artifact.source_row_labels.size() == 4u, "source row labels missing")) return 15;
    for (std::size_t out_row = 0; out_row < artifact.source_row_indices.size(); ++out_row) {
        const std::uint64_t src_row = artifact.source_row_indices[out_row];
        const ct::ptr_t src_begin = source.majorPtr[src_row], src_end = source.majorPtr[src_row + 1u];
        const ct::ptr_t out_begin = artifact.matrix.majorPtr[out_row], out_end = artifact.matrix.majorPtr[out_row + 1u];
        if (!require(src_end - src_begin == out_end - out_begin, "rebuilt row nnz mismatch")) return 16;
        for (ct::ptr_t slot = 0; slot < src_end - src_begin; ++slot) {
            if (!require(source.minorIdx[src_begin + slot] == artifact.matrix.minorIdx[out_begin + slot],
                         "rebuilt column mismatch")) return 17;
            if (!require(close_f32(__half2float(source.val[src_begin + slot]),
                                   __half2float(artifact.matrix.val[out_begin + slot])),
                         "rebuilt value mismatch")) return 18;
        }
    }
    cm::clear(&source);
    return 0;
}

int test_sharded_rebuild() {
    cm::compressed source;
    cd::owned_dataset_artifact artifact;
    cellshard::sharded<cellshard::sparse::compressed> sharded;
    cellshard::sparse::compressed *parts[] = {&source};
    unsigned long partition_offsets[] = {0u, 5u};
    unsigned long partition_rows[] = {5u};
    unsigned long partition_nnz[] = {9u};
    unsigned long partition_aux[] = {0u};
    unsigned long shard_offsets[] = {0u, 5u};
    unsigned long shard_parts[] = {0u, 1u};
    const char *labels[] = {"late", "early", "late", "early", "late"};
    cd::stratified_downsample_request req;
    std::string error;
    cm::init(&source);
    cellshard::init(&sharded);
    if (!require(fill_source(&source), "failed to allocate sharded source matrix")) return 20;
    sharded.rows = 5u;
    sharded.cols = 4u;
    sharded.nnz = 9u;
    sharded.num_partitions = 1u;
    sharded.partition_capacity = 1u;
    sharded.parts = parts;
    sharded.partition_offsets = partition_offsets;
    sharded.partition_rows = partition_rows;
    sharded.partition_nnz = partition_nnz;
    sharded.partition_aux = partition_aux;
    sharded.num_shards = 1u;
    sharded.shard_capacity = 1u;
    sharded.shard_offsets = shard_offsets;
    sharded.shard_parts = shard_parts;
    req.labels = {labels, 5u};
    req.max_rows_per_stratum = 1u;
    req.seed = 17u;
    const cd::dataset_matrix_handle handle = cd::make_cellshard_sharded_compressed_handle(&sharded);
    if (!require(cd::build_stratified_downsample(handle, req, &artifact, &error), error.c_str())) {
        cm::clear(&source);
        return 21;
    }
    if (!require(artifact.matrix.rows == 2u && artifact.matrix.cols == 4u, "sharded rebuild shape mismatch")) return 22;
    if (!require(artifact.row_groups.size() == 2u, "sharded rebuild group count mismatch")) return 23;
    cm::clear(&source);
    return 0;
}

int main() {
    int rc = test_stratified_plan();
    if (rc != 0) return rc;
    rc = test_in_memory_rebuild();
    if (rc != 0) return rc;
    rc = test_sharded_rebuild();
    if (rc != 0) return rc;
    return 0;
}
