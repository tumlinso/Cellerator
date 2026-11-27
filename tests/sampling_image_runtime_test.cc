#include <Cellerator/geometry/discovery/sampling_materialization.hh>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace cs = ::cellerator::geometry::discovery::sampling;
namespace cm = ::cellerator::tensor::base;

int main() {
    const std::uint64_t row_nnz[] = {0u, 1u, 2u, 8u, 8u, 20u};
    cs::density_sample_spec spec;
    spec.seed = 19u;
    spec.split_name = "image-contract";
    spec.requested_strata = 3u;
    spec.requested_row_count = 4u;
    cs::sample_plan plan;
    std::string error;
    if (!cs::build_density_sample_plan(6u, {row_nnz, 6u}, spec, {}, &plan, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::vector<unsigned char> selection_storage(cs::sample_selection_image_bytes(plan));
    cs::sample_selection_view selection;
    if (!cs::encode_sample_selection_image(
            plan, {selection_storage.data(), selection_storage.size(), {}}, &selection, &error)) {
        std::cerr << error << '\n';
        return 2;
    }
    if (selection.header->selected_rows != 4u || selection.header->stratum_count != 3u
        || selection.header->common.identity == 0u || selection.selected_strata == nullptr
        || selection.strata == nullptr) return 3;
    cs::sample_selection_view replay;
    if (!cs::resolve_sample_selection_image(
            {selection_storage.data(), selection_storage.size(), {}}, &replay, &error)
        || replay.header->common.identity != selection.header->common.identity) return 4;

    const std::int64_t row_ptr[] = {0, 0, 1, 3, 4, 6, 7};
    const std::int64_t columns[] = {0, 0, 1, 2, 0, 2, 1};
    const cm::CompressedStructureView<std::int64_t> source{
        6u, 3u, 7u, cm::CompressedAxis::row, row_ptr, columns};
    std::uint64_t selected_nnz = 0u;
    for (std::uint64_t i = 0u; i < selection.header->selected_rows; ++i) {
        const std::uint64_t row = selection.selected_global_rows[i];
        selected_nnz += row_ptr[row + 1u] - row_ptr[row];
    }
    std::vector<unsigned char> csr_storage(
        cs::sampled_csr_image_bytes(selection.header->selected_rows, selected_nnz));
    cs::sampled_csr_image_view sampled;
    if (!cs::materialize_sampled_csr_image(
            source, selection, {csr_storage.data(), csr_storage.size(), {}}, &sampled, &error)) {
        std::cerr << error << '\n';
        return 5;
    }
    cs::sampled_csr_image_view sampled_replay;
    if (!cs::resolve_sampled_csr_image(
            {csr_storage.data(), csr_storage.size(), {}}, &sampled_replay, &error)
        || sampled_replay.header->sample_selection_identity
            != selection.header->common.identity
        || sampled_replay.row_ptr[sampled_replay.header->sampled_row_count]
            != sampled_replay.header->nnz) return 6;

    auto *damaged = reinterpret_cast<cs::sample_selection_header *>(selection_storage.data());
    damaged->selected_global_rows.byte_offset = damaged->common.total_bytes;
    if (cs::resolve_sample_selection_image(
            {selection_storage.data(), selection_storage.size(), {}}, &replay, &error)) return 7;
    return 0;
}
