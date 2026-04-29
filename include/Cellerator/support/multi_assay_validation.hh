#pragma once

#include "../../../extern/CellShard/include/CellShard/io/multi_assay.hh"

#include <Cellerator/support/assay_validation.hh>

#include "cudaBioTypes.hh"

#include <cstdint>

namespace cellerator::semantics {

inline std::uint32_t semantic_code(const bioT::Modality value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const bioT::ObservationUnit value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const lgo::genomics::feature_kind value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const bioT::ValueSemantics value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const bioT::ProcessingState value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const bioT::AxisKind value) {
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t semantic_code(const bioT::IdentifierNamespace value) {
    return static_cast<std::uint32_t>(value);
}

template<typename Assay>
inline cellshard::dataset_assay_semantics make_assay_semantics(const Assay& assay) {
    cellshard::dataset_assay_semantics out{};
    out.modality = semantic_code(assay.metadata.modality);
    out.observation_unit = semantic_code(assay.metadata.observation_unit);
    out.feature_type = semantic_code(assay.matrix.feature_type);
    out.value_semantics = semantic_code(assay.metadata.values);
    out.processing_state = semantic_code(assay.metadata.processing);
    out.row_axis = semantic_code(assay.matrix.rows);
    out.col_axis = semantic_code(assay.matrix.cols);
    out.feature_namespace = semantic_code(assay.matrix.feature_namespace);
    return out;
}

inline bool semantics_match(const cellshard::dataset_assay_semantics& lhs,
                            const cellshard::dataset_assay_semantics& rhs) {
    return lhs.modality == rhs.modality
        && lhs.observation_unit == rhs.observation_unit
        && lhs.feature_type == rhs.feature_type
        && lhs.value_semantics == rhs.value_semantics
        && lhs.processing_state == rhs.processing_state
        && lhs.row_axis == rhs.row_axis
        && lhs.col_axis == rhs.col_axis
        && lhs.feature_namespace == rhs.feature_namespace;
}

template<typename Assay>
inline validation_result validate_assay_semantics(const cellshard::dataset_assay_semantics& stored,
                                                  const Assay& assay) {
    if (!assay.validate()) {
        return failure("bioT assay contract is internally inconsistent");
    }
    if (!cellshard::dataset_assay_semantics_valid(&stored)) {
        return failure("stored assay semantics use an invalid matrix orientation");
    }
    if (!semantics_match(stored, make_assay_semantics(assay))) {
        return failure("stored assay semantics do not match the bioT assay contract");
    }
    return success();
}

template<typename Assay>
inline validation_result validate_assay_view(const cellshard::dataset_assay_view& stored,
                                             const Assay& assay) {
    const auto semantics_check = validate_assay_semantics(stored.semantics, assay);
    if (!semantics_check) return semantics_check;
    if (stored.rows != expected_rows(assay)) {
        return failure("stored assay row count does not match the bioT matrix orientation");
    }
    if (stored.cols != expected_cols(assay)) {
        return failure("stored assay column count does not match the bioT matrix orientation");
    }
    if (!cellshard::dataset_validate_assay_row_map(&stored.row_map)) {
        return failure("stored assay row map is not internally consistent");
    }
    return success();
}

inline validation_result validate_multiome_pairing(const cellshard::dataset_pairing_view& stored,
                                                   const bioT::MultiomeLinkage& multiome) {
    if (!multiome.validate()) {
        return failure("bioT multiome linkage is internally inconsistent");
    }
    const std::uint32_t expected_pairing = static_cast<std::uint32_t>(multiome.pairing);
    if (stored.pairing != expected_pairing) {
        return failure("stored pairing kind does not match the bioT multiome linkage");
    }
    if (!cellshard::dataset_validate_pairing_view(&stored)) {
        return failure("stored paired assay row maps are not execution-valid");
    }
    return success();
}

} // namespace cellerator::semantics
