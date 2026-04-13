#pragma once

#include "cudaBioTypes.hh"

#include "../../extern/CellShard/src/formats/blocked_ell.cuh"
#include "../../extern/CellShard/src/formats/compressed.cuh"

#include <string>

namespace cellerator::semantics {

struct validation_result {
    bool ok = true;
    std::string message;

    BIOT_NODISCARD explicit operator bool() const {
        return ok;
    }
};

inline validation_result success() {
    return {};
}

inline validation_result failure(const std::string& message) {
    return {false, message};
}

template<typename Assay>
inline validation_result validate_assay(const Assay& assay) {
    if (!assay.validate()) {
        return failure("bioT assay contract is internally inconsistent");
    }
    if (assay.matrix.rows == assay.matrix.cols) {
        return failure("assay matrix semantics must map one axis to observations and the other to features");
    }
    return success();
}

template<typename Assay>
inline lgo::core::u32 expected_rows(const Assay& assay) {
    return assay.matrix.rows == bioT::AxisKind::observations
        ? assay.observations.extent()
        : assay.features.extent();
}

template<typename Assay>
inline lgo::core::u32 expected_cols(const Assay& assay) {
    return assay.matrix.cols == bioT::AxisKind::features
        ? assay.features.extent()
        : assay.observations.extent();
}

template<typename Assay>
inline validation_result validate_cellshard_matrix(const cellshard::sparse::compressed& matrix, const Assay& assay) {
    const auto assay_check = validate_assay(assay);
    if (!assay_check) return assay_check;

    const auto expected_axis = assay.matrix.rows == bioT::AxisKind::observations
        ? cellshard::sparse::compressed_by_row
        : cellshard::sparse::compressed_by_col;
    if (matrix.axis != expected_axis) {
        return failure("CellShard compressed axis does not match the assay matrix orientation");
    }

    const auto rows = expected_rows(assay);
    const auto cols = expected_cols(assay);
    if (rows != 0 && matrix.rows != rows) {
        return failure("CellShard compressed row count does not match the assay contract");
    }
    if (cols != 0 && matrix.cols != cols) {
        return failure("CellShard compressed column count does not match the assay contract");
    }
    return success();
}

template<typename Assay>
inline validation_result validate_cellshard_matrix(const cellshard::sparse::blocked_ell& matrix, const Assay& assay) {
    const auto assay_check = validate_assay(assay);
    if (!assay_check) return assay_check;

    if (assay.matrix.rows != bioT::AxisKind::observations
        || assay.matrix.cols != bioT::AxisKind::features) {
        return failure("Blocked-ELL validation requires observation-major assay semantics");
    }

    const auto rows = expected_rows(assay);
    const auto cols = expected_cols(assay);
    if (rows != 0 && matrix.rows != rows) {
        return failure("CellShard blocked-ELL row count does not match the assay contract");
    }
    if (cols != 0 && matrix.cols != cols) {
        return failure("CellShard blocked-ELL column count does not match the assay contract");
    }
    return success();
}

} // namespace cellerator::semantics
