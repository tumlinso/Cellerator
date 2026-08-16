#include <Cellerator/compute/sampling_materialization.hh>

#include <CellShard/io/csh5/api.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace cm = cellerator::matrix;
namespace cs = cellerator::compute::sampling;
namespace ct = cellerator::types;

int require(bool ok, const char *label) {
    if (!ok) std::fprintf(stderr, "%s\n", label);
    return ok ? 1 : 0;
}

bool fill_source(cm::compressed *matrix) {
    cm::init(matrix, 6u, 5u, 7u, cm::compressed_by_row);
    if (!cm::allocate(matrix)) return false;
    const ct::ptr_t row_ptr[] = {0u, 2u, 2u, 3u, 5u, 5u, 7u};
    const ct::idx_t column_indices[] = {0u, 4u, 2u, 1u, 3u, 0u, 2u};
    for (std::size_t i = 0u; i < 7u; ++i) matrix->majorPtr[i] = row_ptr[i];
    for (std::size_t i = 0u; i < 7u; ++i) {
        matrix->minorIdx[i] = column_indices[i];
        matrix->val[i] = __float2half((float) i + 1.0f);
    }
    return true;
}

bool build_exact_plan(std::uint64_t total_rows,
                      std::uint64_t requested_rows,
                      const char *split_name,
                      cs::sample_plan *out,
                      std::string *error) {
    cs::sample_spec spec;
    cs::cell_identity_view identities;
    spec.mode = cs::selection_mode::exact_lowest_hash;
    spec.seed = 31337u;
    spec.split_name = split_name;
    spec.requested_row_count = requested_rows;
    return cs::build_sample_plan(total_rows, spec, identities, out, error);
}

bool matches_complete_source_rows(const cm::compressed &source,
                                  const cs::sampled_csr_structure_view &sampled) {
    if (sampled.row_ptr == nullptr || sampled.provenance == nullptr) return false;
    if (sampled.sampled_row_count != sampled.provenance->selected_rows) return false;
    for (std::size_t position = 0u; position < sampled.sampled_row_count; ++position) {
        const std::uint64_t row = sampled.sampled_position_to_global_row[position];
        const ct::ptr_t source_begin = source.majorPtr[row], source_end = source.majorPtr[row + 1u];
        const ct::ptr_t sample_begin = sampled.row_ptr[position], sample_end = sampled.row_ptr[position + 1u];
        if (source_end - source_begin != sample_end - sample_begin) return false;
        for (ct::ptr_t slot = 0u; slot < source_end - source_begin; ++slot) {
            if (source.minorIdx[source_begin + slot] != sampled.column_indices[sample_begin + slot]) return false;
        }
    }
    return sampled.row_ptr[sampled.sampled_row_count] == sampled.nnz;
}

int test_in_memory_complete_rows_and_ordering() {
    static_assert(std::is_same<
                      decltype(std::declval<const cs::owned_sampled_csr_structure &>().sampling_provenance()),
                      const cs::sample_provenance &>::value,
                  "sampling provenance must be exposed read-only");
    static_assert(std::is_same<
                      std::remove_cv_t<std::remove_pointer_t<
                          decltype(std::declval<cs::sampled_csr_structure_view>().column_indices)>>,
                      ct::idx_t>::value,
                  "sampled CSR columns must use Cellerator's canonical index type");

    cm::compressed source;
    cs::sample_plan plan, unsorted;
    cs::owned_sampled_csr_structure bundle;
    std::string error;
    cm::init(&source);
    if (!require(fill_source(&source), "failed to allocate in-memory CSR fixture")) return 1;
    if (!require(build_exact_plan(source.rows, 64u, "in-memory-all", &plan, &error), error.c_str())) return 2;
    if (!require(plan.global_row_indices.size() == source.rows,
                 "undersized dataset was not sampled completely")) return 3;

    unsorted = plan;
    std::reverse(unsorted.global_row_indices.begin(), unsorted.global_row_indices.end());
    std::reverse(unsorted.identity_hashes.begin(), unsorted.identity_hashes.end());
    if (!require(cs::materialize_sampled_csr_structure(&source, unsorted, &bundle, &error), error.c_str())) return 4;
    const cs::sampled_csr_structure_view view = bundle.view();
    if (!require(view.sampled_row_count == 6u && view.gene_count == 5u && view.nnz == 7u,
                 "sampled in-memory CSR dimensions mismatch")) return 5;
    if (!require(view.row_ptr != nullptr && view.column_indices != nullptr
                 && view.sampled_position_to_global_row != nullptr,
                 "sampled in-memory CSR arrays are missing")) return 6;
    for (std::uint64_t position = 0u; position < view.sampled_row_count; ++position) {
        if (!require(view.sampled_position_to_global_row[position] == position,
                     "unsorted request did not canonicalize sampled positions")) return 7;
    }
    if (!require(view.row_ptr[1] == view.row_ptr[2]
                 && view.row_ptr[4] == view.row_ptr[5],
                 "empty source rows were not preserved")) return 8;
    if (!require(matches_complete_source_rows(source, view),
                 "sampled structural rows differ from complete source rows")) return 9;
    if (!require(view.provenance == &bundle.sampling_provenance()
                 && view.provenance->seed == plan.provenance.seed
                 && view.provenance->requested_row_count == 64u
                 && view.provenance->selected_rows == 6u
                 && view.provenance->split_name == "in-memory-all",
                 "sample provenance was not preserved immutably")) return 10;
    cm::clear(&source);
    return 0;
}

int test_zero_rows_and_input_validation() {
    cm::compressed source;
    cs::sample_plan all_rows, zero_rows, invalid;
    cs::owned_sampled_csr_structure bundle;
    std::string error;
    cm::init(&source);
    if (!require(fill_source(&source), "failed to allocate validation CSR fixture")) return 20;
    if (!require(build_exact_plan(source.rows, 64u, "validation-all", &all_rows, &error), error.c_str())) return 21;
    if (!require(build_exact_plan(source.rows, 0u, "zero-rows", &zero_rows, &error), error.c_str())) return 22;
    if (!require(cs::materialize_sampled_csr_structure(&source, zero_rows, &bundle, &error), error.c_str())) return 23;
    cs::sampled_csr_structure_view view = bundle.view();
    if (!require(view.sampled_row_count == 0u && view.gene_count == source.cols && view.nnz == 0u
                 && view.row_ptr != nullptr && view.row_ptr[0] == 0u
                 && view.column_indices == nullptr && view.sampled_position_to_global_row == nullptr,
                 "zero-row sampled CSR representation is invalid")) return 24;

    invalid = all_rows;
    invalid.global_row_indices[0] = source.rows;
    invalid.identity_hashes[0] = cs::hash_global_row_index(source.rows, invalid.provenance.seed);
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, invalid, &bundle, &error),
                 "out-of-range sampled row was accepted")) return 25;

    invalid = all_rows;
    invalid.global_row_indices[1] = invalid.global_row_indices[0];
    invalid.identity_hashes[1] = invalid.identity_hashes[0];
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, invalid, &bundle, &error),
                 "duplicate sampled row was accepted")) return 26;

    invalid = all_rows;
    invalid.provenance.selected_rows = UINT64_MAX;
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, invalid, &bundle, &error),
                 "overflowed provenance selected_rows was accepted")) return 27;

    const ct::ptr_t saved_second = source.majorPtr[2u];
    source.majorPtr[2u] = source.nnz + 1u;
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, all_rows, &bundle, &error),
                 "row pointer beyond nnz was accepted")) return 28;
    source.majorPtr[2u] = saved_second;

    const ct::ptr_t saved_third = source.majorPtr[3u];
    source.majorPtr[3u] = 1u;
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, all_rows, &bundle, &error),
                 "non-monotonic row pointers were accepted")) return 29;
    source.majorPtr[3u] = saved_third;

    const ct::ptr_t saved_terminal = source.majorPtr[source.rows];
    source.majorPtr[source.rows] = source.nnz - 1u;
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, all_rows, &bundle, &error),
                 "incorrect terminal row pointer was accepted")) return 30;
    source.majorPtr[source.rows] = saved_terminal;

    const ct::idx_t saved_column = source.minorIdx[0u];
    source.minorIdx[0u] = source.cols;
    error.clear();
    if (!require(!cs::materialize_sampled_csr_structure(&source, all_rows, &bundle, &error),
                 "out-of-range column index was accepted")) return 31;
    source.minorIdx[0u] = saved_column;
    cm::clear(&source);
    return 0;
}

struct cellshard_fixture {
    std::string path;
    cellshard::sparse::blocked_ell part;
    cellshard::bucketed_blocked_ell_shard shard;

    cellshard_fixture() {
        cellshard::sparse::init(&part);
        cellshard::init(&shard);
    }
    ~cellshard_fixture() {
        cellshard::clear(&shard);
        cellshard::sparse::clear(&part);
        if (!path.empty()) std::remove(path.c_str());
    }
};

bool create_cellshard_fixture(cellshard_fixture *fixture) {
    cellshard::bucketed_blocked_ell_partition bucket;
    cellshard::init(&bucket);
    char path[] = "/tmp/cellerator_sample_structureXXXXXX.csh5";
    const int fd = ::mkstemps(path, 5);
    if (fd < 0) return false;
    ::close(fd);
    std::remove(path);
    fixture->path = path;

    cellshard::sparse::init(&fixture->part, 2u, 4u, 4u, 2u, 4u);
    if (!cellshard::sparse::allocate(&fixture->part)) return false;
    fixture->part.blockColIdx[0] = 0u;
    fixture->part.blockColIdx[1] = 1u;
    fixture->part.val[0] = __float2half(1.0f);
    fixture->part.val[1] = __float2half(0.0f);
    fixture->part.val[2] = __float2half(2.0f);
    fixture->part.val[3] = __float2half(0.0f);
    fixture->part.val[4] = __float2half(0.0f);
    fixture->part.val[5] = __float2half(3.0f);
    fixture->part.val[6] = __float2half(0.0f);
    fixture->part.val[7] = __float2half(4.0f);

    const std::uint64_t partition_rows[] = {2u}, partition_nnz[] = {4u};
    const std::uint32_t partition_axes[] = {0u};
    const std::uint64_t partition_aux[] = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    const std::uint64_t partition_offsets[] = {0u, 2u}, shard_offsets[] = {0u, 2u};
    const std::uint32_t partition_dataset_ids[] = {0u}, partition_codec_ids[] = {0u};
    cellshard::dataset_codec_descriptor codec{};
    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    const cellshard::dataset_layout_view layout{
        2u, 4u, 4u, 1u, 1u,
        partition_rows, partition_nnz, partition_axes, partition_aux, partition_offsets,
        partition_dataset_ids, partition_codec_ids, shard_offsets, &codec, 1u
    };
    if (!cellshard::build_bucketed_blocked_ell_partition(&bucket, &fixture->part, 1u, nullptr)) {
        return false;
    }
    bucket.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    bucket.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    fixture->shard.rows = 2u;
    fixture->shard.cols = 4u;
    fixture->shard.nnz = 4u;
    fixture->shard.partition_count = 1u;
    fixture->shard.partitions = (cellshard::bucketed_blocked_ell_partition *)
        std::calloc(1u, sizeof(cellshard::bucketed_blocked_ell_partition));
    fixture->shard.partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
    fixture->shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    fixture->shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    if (bucket.exec_to_canonical_cols == nullptr || bucket.canonical_to_exec_cols == nullptr
        || fixture->shard.partitions == nullptr || fixture->shard.partition_row_offsets == nullptr
        || fixture->shard.exec_to_canonical_cols == nullptr
        || fixture->shard.canonical_to_exec_cols == nullptr) {
        cellshard::clear(&bucket);
        return false;
    }
    for (std::uint32_t column = 0u; column < 4u; ++column) {
        bucket.exec_to_canonical_cols[column] = column;
        bucket.canonical_to_exec_cols[column] = column;
        fixture->shard.exec_to_canonical_cols[column] = column;
        fixture->shard.canonical_to_exec_cols[column] = column;
    }
    fixture->shard.partitions[0] = bucket;
    cellshard::init(&bucket);
    fixture->shard.partition_row_offsets[0] = 0u;
    fixture->shard.partition_row_offsets[1] = 2u;
    return cellshard::create_dataset_optimized_blocked_ell_h5(
               fixture->path.c_str(), &layout, nullptr, nullptr) != 0
        && cellshard::append_bucketed_blocked_ell_shard_h5(
               fixture->path.c_str(), 0ul, &fixture->shard) != 0;
}

int test_cellshard_complete_rows() {
    cellshard_fixture fixture;
    cs::sample_spec spec;
    cs::sample_plan plan;
    cs::owned_sampled_csr_structure bundle;
    std::string error;
    if (!require(create_cellshard_fixture(&fixture), "failed to create CellShard structural fixture")) return 40;
    spec.mode = cs::selection_mode::exact_lowest_hash;
    spec.seed = 919u;
    spec.split_name = "cellshard-all";
    spec.requested_row_count = 64u;
    if (!require(cs::build_cellshard_sample_plan(fixture.path.c_str(), spec, &plan, &error), error.c_str())) return 41;
    std::reverse(plan.global_row_indices.begin(), plan.global_row_indices.end());
    std::reverse(plan.identity_hashes.begin(), plan.identity_hashes.end());
    if (!require(cs::materialize_cellshard_sampled_csr_structure(
                     fixture.path.c_str(), plan, &bundle, &error), error.c_str())) return 42;
    const cs::sampled_csr_structure_view view = bundle.view();
    if (!require(view.sampled_row_count == 2u && view.gene_count == 4u && view.nnz == 4u,
                 "CellShard sampled CSR dimensions mismatch")) return 43;
    const ct::ptr_t expected_ptr[] = {0u, 2u, 4u};
    const ct::idx_t expected_indices[] = {0u, 2u, 1u, 3u};
    for (std::size_t i = 0u; i < 3u; ++i) {
        if (!require(view.row_ptr[i] == expected_ptr[i], "CellShard sampled row pointer mismatch")) return 44;
    }
    for (std::size_t i = 0u; i < 4u; ++i) {
        if (!require(view.column_indices[i] == expected_indices[i],
                     "CellShard sampled complete row contents mismatch")) return 45;
    }
    if (!require(view.sampled_position_to_global_row[0] == 0u
                 && view.sampled_position_to_global_row[1] == 1u,
                 "CellShard sampled positions are not canonical global-row order")) return 46;
    if (!require(view.provenance->requested_row_count == 64u
                 && view.provenance->selected_rows == 2u
                 && view.provenance->cell_identity == cs::cell_identity_kind::global_row_index,
                 "CellShard sampled provenance mismatch")) return 47;
    spec.split_name = "cellshard-zero";
    spec.requested_row_count = 0u;
    error.clear();
    if (!require(cs::build_cellshard_sample_plan(fixture.path.c_str(), spec, &plan, &error), error.c_str())) return 48;
    if (!require(cs::materialize_cellshard_sampled_csr_structure(
                     fixture.path.c_str(), plan, &bundle, &error), error.c_str())) return 49;
    const cs::sampled_csr_structure_view zero_view = bundle.view();
    if (!require(zero_view.sampled_row_count == 0u && zero_view.gene_count == 4u
                 && zero_view.nnz == 0u && zero_view.row_ptr != nullptr
                 && zero_view.row_ptr[0] == 0u && zero_view.column_indices == nullptr
                 && zero_view.sampled_position_to_global_row == nullptr,
                 "CellShard zero-row sampled CSR representation is invalid")) return 50;
    return 0;
}

int main() {
    int rc = test_in_memory_complete_rows_and_ordering();
    if (rc != 0) return rc;
    rc = test_zero_rows_and_input_validation();
    if (rc != 0) return rc;
    return test_cellshard_complete_rows();
}
