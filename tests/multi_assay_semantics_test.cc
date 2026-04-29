#include <Cellerator/support/multi_assay_validation.hh>

#include <CellShard/io/cshard/spec.hh>
#include <CellShard/io/multi_assay.hh>

#include <cassert>
#include <cstdint>
#include <vector>

namespace cs = ::cellshard;
namespace sem = ::cellerator::semantics;

namespace {

cs::dataset_assay_row_map_view make_row_map(const std::vector<std::uint32_t>& global_to_assay,
                                            const std::vector<std::uint32_t>& assay_to_global) {
    cs::dataset_assay_row_map_view out{};
    out.global_observation_count = static_cast<std::uint32_t>(global_to_assay.size());
    out.assay_row_count = static_cast<std::uint32_t>(assay_to_global.size());
    out.global_to_assay_row = global_to_assay.data();
    out.assay_row_to_global = assay_to_global.data();
    return out;
}

void test_exact_rna_atac_pairing() {
    bioT::RnaAssay rna;
    rna.observations.count = 3u;
    rna.features.count = 4u;

    bioT::AtacAssay atac;
    atac.observations.count = 3u;
    atac.features.count = 8u;

    const std::vector<std::uint32_t> exact = {0u, 1u, 2u};
    cs::dataset_assay_view assays[2]{};
    assays[0].assay_id = "rna";
    assays[0].semantics = sem::make_assay_semantics(rna);
    assays[0].rows = 3u;
    assays[0].cols = 4u;
    assays[0].feature_order_hash = 11u;
    assays[0].row_map = make_row_map(exact, exact);
    assays[1].assay_id = "atac";
    assays[1].semantics = sem::make_assay_semantics(atac);
    assays[1].rows = 3u;
    assays[1].cols = 8u;
    assays[1].feature_order_hash = 22u;
    assays[1].row_map = make_row_map(exact, exact);

    bioT::MultiomeLinkage multiome;
    multiome.rna = rna;
    multiome.atac = atac;
    multiome.pairing = bioT::PairingKind::exact_observation;
    multiome.has_rna = {1u, 1u, 1u};
    multiome.has_atac = {1u, 1u, 1u};

    cs::dataset_pairing_view pairing{};
    pairing.pairing = cs::dataset_pairing_exact_observation;
    pairing.assay_count = 2u;
    pairing.assays = assays;
    assert(sem::validate_assay_view(assays[0], rna));
    assert(sem::validate_assay_view(assays[1], atac));
    assert(sem::validate_multiome_pairing(pairing, multiome));

    std::uint32_t rna_row = cs::dataset_assay_invalid_row, atac_row = cs::dataset_assay_invalid_row;
    assert(cs::dataset_resolve_paired_rows(&assays[0].row_map, &assays[1].row_map, 2u, &rna_row, &atac_row));
    assert(rna_row == 2u && atac_row == 2u);
}

void test_partial_pairing_keeps_missing_modality() {
    bioT::RnaAssay rna;
    rna.observations.count = 3u;
    rna.features.count = 4u;

    bioT::AtacAssay atac;
    atac.observations.count = 2u;
    atac.features.count = 8u;

    const std::vector<std::uint32_t> rna_global_to_assay = {0u, 1u, 2u};
    const std::vector<std::uint32_t> rna_assay_to_global = {0u, 1u, 2u};
    const std::vector<std::uint32_t> atac_global_to_assay = {
        0u,
        cs::dataset_assay_invalid_row,
        1u
    };
    const std::vector<std::uint32_t> atac_assay_to_global = {0u, 2u};

    cs::dataset_assay_view assays[2]{};
    assays[0].assay_id = "rna";
    assays[0].semantics = sem::make_assay_semantics(rna);
    assays[0].rows = 3u;
    assays[0].cols = 4u;
    assays[0].row_map = make_row_map(rna_global_to_assay, rna_assay_to_global);
    assays[1].assay_id = "atac";
    assays[1].semantics = sem::make_assay_semantics(atac);
    assays[1].rows = 2u;
    assays[1].cols = 8u;
    assays[1].row_map = make_row_map(atac_global_to_assay, atac_assay_to_global);

    bioT::MultiomeLinkage multiome;
    multiome.rna = rna;
    multiome.atac = atac;
    multiome.pairing = bioT::PairingKind::partial_observation;
    multiome.has_rna = {1u, 1u, 1u};
    multiome.has_atac = {1u, 0u, 1u};
    assert(multiome.validate());

    cs::dataset_pairing_view pairing{};
    pairing.pairing = cs::dataset_pairing_partial_observation;
    pairing.assay_count = 2u;
    pairing.assays = assays;
    assert(sem::validate_multiome_pairing(pairing, multiome));

    std::uint32_t rna_row = 7u, atac_row = 7u;
    assert(cs::dataset_resolve_paired_rows(&assays[0].row_map, &assays[1].row_map, 1u, &rna_row, &atac_row));
    assert(rna_row == 1u);
    assert(atac_row == cs::dataset_assay_invalid_row);
}

void test_malformed_row_map_rejected() {
    const std::vector<std::uint32_t> global_to_assay = {0u, 1u};
    const std::vector<std::uint32_t> assay_to_global = {0u, 0u};
    const cs::dataset_assay_row_map_view row_map = make_row_map(global_to_assay, assay_to_global);
    assert(!cs::dataset_validate_assay_row_map(&row_map));
}

void test_archive_descriptors_are_fixed_width() {
    static_assert(sizeof(cs::cshard::spec::header) == 256u);
    static_assert(sizeof(cs::cshard::spec::assay_descriptor) % 8u == 0u);
    static_assert(sizeof(cs::cshard::spec::pairing_descriptor) % 8u == 0u);
    static_assert(sizeof(cs::cshard::spec::assay_pack_manifest_descriptor) % 8u == 0u);

    cs::cshard::spec::assay_descriptor rna{};
    rna.modality = cs::dataset_modality_scrna;
    rna.observation_unit = cs::dataset_observation_cell;
    rna.feature_type = cs::dataset_feature_gene;
    rna.value_semantics = cs::dataset_values_raw_counts;
    rna.processing_state = cs::dataset_processing_raw;
    rna.row_axis = cs::dataset_axis_observations;
    rna.col_axis = cs::dataset_axis_features;
    rna.global_observation_count = 3u;
    rna.assay_row_count = 3u;
    rna.feature_count = 4u;
    assert(rna.modality == static_cast<std::uint32_t>(bioT::Modality::scrna));
    assert(rna.feature_type == static_cast<std::uint32_t>(lgo::genomics::feature_kind::gene));
}

} // namespace

int main() {
    test_exact_rna_atac_pairing();
    test_partial_pairing_keeps_missing_modality();
    test_malformed_row_map_rejected();
    test_archive_descriptors_are_fixed_width();
    return 0;
}
