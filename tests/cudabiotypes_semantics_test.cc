#include "../src/support/assay_validation.hh"

#include <cassert>

int main() {
    bioT::RnaAssay rna;
    rna.observations.count = 3;
    rna.features.count = 4;
    assert(rna.validate());

    cellshard::sparse::compressed csr{};
    cellshard::sparse::init(&csr, 3, 4, 0, cellshard::sparse::compressed_by_row);
    auto ok = cellerator::semantics::validate_cellshard_matrix(csr, rna);
    assert(ok);

    cellshard::sparse::compressed csc{};
    cellshard::sparse::init(&csc, 3, 4, 0, cellshard::sparse::compressed_by_col);
    auto wrong_axis = cellerator::semantics::validate_cellshard_matrix(csc, rna);
    assert(!wrong_axis);

    cellshard::sparse::blocked_ell blocked{};
    cellshard::sparse::init(&blocked, 3, 4, 0, 2, 4);
    auto blocked_ok = cellerator::semantics::validate_cellshard_matrix(blocked, rna);
    assert(blocked_ok);

    bioT::AtacAssay atac;
    atac.observations.count = 3;
    atac.features.count = 8;
    atac.matrix.rows = bioT::AxisKind::features;
    atac.matrix.cols = bioT::AxisKind::observations;
    assert(atac.validate());

    cellshard::sparse::compressed atac_csc{};
    cellshard::sparse::init(&atac_csc, 8, 3, 0, cellshard::sparse::compressed_by_col);
    auto atac_ok = cellerator::semantics::validate_cellshard_matrix(atac_csc, atac);
    assert(atac_ok);

    cellshard::sparse::blocked_ell atac_blocked{};
    cellshard::sparse::init(&atac_blocked, 8, 3, 0, 2, 4);
    auto blocked_fail = cellerator::semantics::validate_cellshard_matrix(atac_blocked, atac);
    assert(!blocked_fail);

    bioT::MultiomeLinkage multiome;
    multiome.rna = rna;
    multiome.atac = atac;
    multiome.pairing = bioT::PairingKind::partial_observation;
    multiome.has_rna = {1, 1, 1};
    multiome.has_atac = {1, 0, 1};
    assert(multiome.validate());

    return 0;
}
