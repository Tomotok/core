# Licensed under the EUPL-1.2 or later.
import unittest

import numpy as np
from scipy import sparse

from tomotok.regularisations.derivatives import (
    all_direction_derivative_matrices,
    compensate_matrix,
    laplace_matrix,
    reduce_matrix,
    standard_anisotropic_derivative_matrices,
)
from tomotok.geometry import RegularGrid


class DerivativeHelpersTestCase(unittest.TestCase):
    def setUp(self):
        self.grid = RegularGrid(6, 5, (1.0, 2.2), (-0.5, 0.5))

    def test_reduce_matrix_accepts_2d_mask(self):
        mat = sparse.diags_array([1.0, 2.0, 3.0, 4.0], format='csc')
        mask = np.array([[True, False], [True, False]])

        reduced = reduce_matrix(mat, mask)

        self.assertEqual(reduced.shape, (2, 2))
        np.testing.assert_allclose(reduced.toarray(), np.diag([1.0, 3.0]))

    def test_reduce_matrix_rejects_invalid_mask_rank(self):
        mat = sparse.eye_array(4, format='csc')
        bad_mask = np.ones((2, 2, 1), dtype=bool)

        self.assertRaises(ValueError, reduce_matrix, mat, bad_mask)

    def test_reduce_matrix_rejects_size_mismatch(self):
        mat = sparse.eye_array(4, format='csc')
        bad_mask = np.array([True, False, True], dtype=bool)

        self.assertRaises(ValueError, reduce_matrix, mat, bad_mask)

    def test_compensate_matrix_zeroes_row_sum_for_full_compensation(self):
        mat = sparse.csr_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

        compensated = compensate_matrix(mat, compensation_fraction=1.0)
        row_sum = np.asarray(compensated.sum(axis=1)).flatten()

        np.testing.assert_allclose(row_sum, np.zeros(2), atol=1e-12)

    def test_all_direction_derivative_matrices_returns_eight_matrices(self):
        matrices = all_direction_derivative_matrices(self.grid, scheme='forward')

        self.assertEqual(len(matrices), 8)
        for dmat in matrices:
            self.assertEqual(dmat.shape, (self.grid.size, self.grid.size))

    def test_standard_anisotropic_derivative_matrices_returns_four_matrices(self):
        rr, zz = self.grid.center_mesh
        flux = rr**2 + zz**2

        matrices = standard_anisotropic_derivative_matrices(self.grid, flux)

        self.assertEqual(len(matrices), 4)
        for dmat in matrices:
            self.assertEqual(dmat.shape, (self.grid.size, self.grid.size))

    def test_laplace_matrix_mask_reduces_size(self):
        mask = np.ones(self.grid.shape, dtype=bool)
        mask[0, 0] = False
        mask[1, 1] = False

        lmat = laplace_matrix(self.grid, mask=mask, compensate_edges=True)

        self.assertEqual(lmat.shape, (mask.sum(), mask.sum()))