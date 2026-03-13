# Licensed under the EUPL-1.2 or later.
import unittest

import numpy as np
from scipy import sparse

from tomotok.core.regularisation import regularisation_matrix


class RegularisationMatrixTestCase(unittest.TestCase):
    def test_single_derivative_defaults_to_identity_weighting(self):
        dmat = sparse.eye_array(3, format='csc')
        reg = regularisation_matrix(dmat)

        self.assertEqual(reg.shape, (3, 3))
        np.testing.assert_allclose(reg.toarray(), np.eye(3))

    def test_multiple_derivatives_with_weights(self):
        dmat_1 = sparse.eye_array(3, format='csc')
        dmat_2 = 2 * sparse.eye_array(3, format='csc')
        node_weights = np.array([1.0, 2.0, 3.0])

        reg = regularisation_matrix(
            [dmat_1, dmat_2],
            derivative_weights=[1.0, 3.0],
            node_weights=node_weights,
        )

        # (1/4) * N + (3/4) * 4N = (13/4) * N
        expected = np.diag(13 / 4 * node_weights)
        np.testing.assert_allclose(reg.toarray(), expected)

    def test_derivative_weights_length_mismatch_raises(self):
        dmat = sparse.eye_array(2, format='csc')

        self.assertRaises(
            ValueError,
            regularisation_matrix,
            [dmat, dmat],
            derivative_weights=[1.0],
        )

    def test_non_iterable_derivative_weights_raises(self):
        dmat = sparse.eye_array(2, format='csc')

        self.assertRaises(
            TypeError,
            regularisation_matrix,
            [dmat, dmat],
            derivative_weights=1,
        )