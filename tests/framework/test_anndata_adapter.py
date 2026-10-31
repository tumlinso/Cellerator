"""Focused source adapter checks; AnnData and SciPy remain optional providers."""
import unittest

import numpy as np

try:
    import anndata
    from scipy import sparse as scipy_sparse
except ImportError:
    anndata = None
    scipy_sparse = None

from cellerator.adapters.anndata import AnnDataView, write_observation
from cellerator.modeling import DynamicalModel


@unittest.skipUnless(anndata is not None and scipy_sparse is not None,
                     "AnnData adapter checks require AnnData and SciPy")
class AnnDataAdapterTests(unittest.TestCase):
    @staticmethod
    def example():
        matrix = scipy_sparse.csr_matrix(
            np.array([[0., 2., 0.], [3., 0., 4.], [0., 5., 6.]])
        )
        adata = anndata.AnnData(matrix)
        adata.obs_names = ["c-a", "c-b", "c-c"]
        adata.var_names = ["g-a", "g-b", "g-c"]
        adata.layers["counts"] = matrix.copy()
        adata.raw = adata
        return adata

    def test_explicit_sparse_source_and_identity_selection_preserve_sparse_rows(self):
        adata = self.example()
        view = AnnDataView.from_adata(
            adata, layer="counts", cell_ids=["c-c", "c-a"], feature_ids=["g-c", "g-a"]
        )
        self.assertEqual(view.source, "layers['counts']")
        self.assertEqual(view.cell_ids, ("c-c", "c-a"))
        self.assertEqual(view.feature_ids, ("g-c", "g-a"))
        self.assertTrue(scipy_sparse.issparse(view.values))
        self.assertEqual(view.values.format, "csr")
        state = view.to_cell_states()
        self.assertTrue(scipy_sparse.issparse(state["c-c"]["measurement"]))
        self.assertEqual(state["c-c"]["measurement"].shape, (1, 2))

    def test_selected_rows_bind_to_model_cells_with_stable_ids(self):
        view = AnnDataView.from_adata(self.example(), cell_ids=["c-b", "c-a"])
        cells = view.to_cells(DynamicalModel(lambda state, _: state))
        self.assertEqual(tuple(cell.id for cell in cells), ("c-b", "c-a"))
        self.assertTrue(scipy_sparse.issparse(cells[0].state["measurement"]))

    def test_raw_selection_and_missing_or_ambiguous_selection_are_explicit(self):
        adata = self.example()
        raw = AnnDataView.from_adata(adata, use_raw=True)
        self.assertEqual(raw.source, "raw.X")
        with self.assertRaises(ValueError):
            AnnDataView.from_adata(adata, layer="counts", use_raw=True)
        with self.assertRaises(KeyError):
            AnnDataView.from_adata(adata, cell_ids=["not-a-cell"])

    def test_write_layer_aligns_both_identity_axes_and_retains_sparse_format(self):
        adata = self.example()
        values = scipy_sparse.csc_matrix(np.array([[9., 0.], [0., 8.]]))
        written = write_observation(
            adata, values, cell_ids=["c-c", "c-a"], feature_ids=["g-c", "g-a"],
            target="layer", key="prediction",
        )
        self.assertTrue(scipy_sparse.isspmatrix_csc(written))
        self.assertEqual(written.shape, (3, 3))
        np.testing.assert_array_equal(
            written.toarray(), [[8., 0., 0.], [0., 0., 0.], [0., 0., 9.]]
        )
        self.assertIs(adata.layers["prediction"], written)

    def test_write_obsm_and_obs_align_rows_by_identity(self):
        adata = self.example()
        result = write_observation(
            adata, [[7., 8.]], cell_ids=["c-b"], target="obsm", key="latent"
        )
        np.testing.assert_array_equal(result, [[0., 0.], [7., 8.], [0., 0.]])
        write_observation(
            adata, [0.25, 0.75], cell_ids=["c-c", "c-a"], target="obs", key="score"
        )
        np.testing.assert_allclose(adata.obs["score"].to_numpy(), [0.75, 0., 0.25])

    def test_layer_export_requires_explicit_feature_identities(self):
        adata = self.example()
        with self.assertRaises(ValueError):
            write_observation(
                adata, [[1., 2., 3.]], cell_ids=["c-a"], target="layer", key="y"
            )


if __name__ == "__main__":
    unittest.main()
