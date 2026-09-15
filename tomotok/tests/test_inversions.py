# Licensed under the EUPL-1.2 or later.
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

import tomotok.core
from tomotok.inversions import CholeskySolver, FixedSelector, MinimumFisherRegularisation, Tikhonov
from tomotok.inversions.base import PearsonSelector, RegularisedInversion, Inversion
from tomotok.inversions.bob import Bob, SparseInvSolver
from tomotok.inversions.lame import FastSelector, GevAlgebraic, SvdAlgebraic


class EchoInversion(RegularisedInversion):
    def invert(self, alpha: float) -> np.ndarray:
        return alpha * self._data


class ConstantInversion(RegularisedInversion):
    def invert(self, alpha: float) -> np.ndarray:
        return np.full(self._gmat.shape[1], alpha)


class BaseInversionTestCase(unittest.TestCase):
    def test_inversion_rejects_invalid_solver(self):
        inversion = Inversion()
        self.assertRaises(ValueError, setattr, inversion, 'solver', object())

    def test_cholesky_solver_solves_dense_and_sparse_inputs(self):
        solver = CholeskySolver()
        a = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])

        dense_solution = solver.solve(a, b)
        sparse_solution = solver.solve(sparse.csr_array(a), sparse.csr_array(b.reshape(-1, 1))).flatten()

        np.testing.assert_allclose(dense_solution, np.linalg.solve(a, b))
        np.testing.assert_allclose(sparse_solution, np.linalg.solve(a, b))

    def test_fixed_selector_returns_fixed_value(self):
        selector = FixedSelector(5.0)

        alpha_1, stats_1 = selector.determine()
        alpha_2, stats_2 = selector.determine()  # same value each call

        self.assertEqual(alpha_1, 5.0)
        self.assertEqual(alpha_2, 5.0)
        self.assertAlmostEqual(stats_1['logalpha'], np.log10(5.0))
        self.assertAlmostEqual(stats_2['logalpha'], np.log10(5.0))

    def test_regularised_inversion_normalises_inputs(self):
        inversion = EchoInversion(regularisation_selector=FixedSelector(0.1))
        data = np.array([2.0, 4.0])
        gmat = np.eye(2)
        regularisation = np.eye(2)

        out, stats = inversion(data, gmat, regularisation, errors=2.0)

        np.testing.assert_allclose(out, np.array([0.1, 0.2]))
        self.assertAlmostEqual(stats['logalpha'], np.log10(0.1))

    def test_regularised_inversion_rejects_error_shape_mismatch(self):
        inversion = EchoInversion(regularisation_selector=FixedSelector(1.0))

        self.assertRaises(
            ValueError,
            inversion,
            np.array([1.0, 2.0]),
            np.eye(2),
            np.eye(2),
            np.array([1.0]),
        )

    def test_pearson_selector_returns_positive_alpha(self):
        selector = PearsonSelector(bounds=(-5, 2), iter_max=25, tolerance=1e-5)
        solver = Tikhonov(regularisation_selector=selector)

        out, stats = solver(np.array([1.0, 2.0]), np.eye(2), np.eye(2), errors=1.0)

        self.assertEqual(out.shape, (2,))
        self.assertGreater(stats['iter_num'], 0)
        self.assertGreater(10 ** stats['logalpha'], 0)


class ConcreteInversionTestCase(unittest.TestCase):
    def test_tikhonov_with_fixed_regularisation(self):
        solver = Tikhonov(regularisation_selector=FixedSelector(1.0))

        out, stats = solver(np.array([1.0, 2.0]), np.eye(2), np.eye(2), errors=1.0)

        np.testing.assert_allclose(out, np.array([0.5, 1.0]))
        self.assertAlmostEqual(stats['logalpha'], 0.0)

    def test_svd_algebraic_on_identity_problem(self):
        solver = SvdAlgebraic(regularisation_selector=FixedSelector(1.0))

        out, stats = solver(np.array([1.0, 2.0]), np.eye(2), np.eye(2), errors=1.0)

        np.testing.assert_allclose(out, np.array([0.5, 1.0]))
        self.assertAlmostEqual(stats['logalpha'], 0.0)

    def test_gev_algebraic_produces_finite_output(self):
        solver = GevAlgebraic(regularisation_selector=FixedSelector(1.0), num=1)
        gmat = sparse.csr_array(np.eye(2))
        regularisation = sparse.csr_array(np.eye(2))

        out, stats = solver(np.array([1.0, 2.0]), gmat, regularisation, errors=1.0)

        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(stats['logalpha'], 0.0)

    def test_fast_selector_methods(self):
        solver = SvdAlgebraic(regularisation_selector=FixedSelector(0.0))
        solver.s = np.array([1.0, 10.0, 100.0])

        mean_alpha, mean_stats = FastSelector('mean').determine(solver)
        logmean_alpha, logmean_stats = FastSelector('logmean').determine(solver)

        self.assertAlmostEqual(mean_alpha, solver.s.mean())
        self.assertAlmostEqual(logmean_alpha, 10.0)
        self.assertEqual(mean_stats['method'], 'mean')
        self.assertEqual(logmean_stats['method'], 'logmean')
        self.assertRaises(ValueError, FastSelector, 'invalid')

    def test_fast_selector_requires_algebraic_solver(self):
        self.assertRaises(TypeError, FastSelector().determine, Tikhonov())

    def test_bob_identity_decomposition_reconstructs_data(self):
        bob = Bob()
        gmat = sparse.csr_array(np.eye(2))
        basis = sparse.csc_array(np.eye(2))
        data = np.array([1.0, 2.0])

        out = bob(data, gmat=gmat, basis=basis)

        np.testing.assert_allclose(out, data)

    def test_bob_save_and_load_decomposition(self):
        bob = Bob()
        gmat = sparse.csr_array(np.eye(2))
        basis = sparse.csc_array(np.eye(2))
        bob.decompose(gmat, basis)
        bob.normalise()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'bob.h5'
            bob.save_decomposition(path)
            loaded = Bob()
            loaded.load_decomposition(path)

        np.testing.assert_allclose(loaded._basis.toarray(), basis.toarray())
        np.testing.assert_allclose(loaded._adjoint_basis.toarray(), bob._adjoint_basis.toarray())
        np.testing.assert_allclose(loaded._norms, bob._norms)

    def test_sparse_inv_solver_identity_rhs_only(self):
        solver = SparseInvSolver()
        a = np.array([[2.0, 0.0], [0.0, 4.0]])

        inverse = solver.solve(a, np.eye(2))

        np.testing.assert_allclose(inverse.toarray(), np.linalg.inv(a))
        self.assertRaises(ValueError, solver.solve, a, np.array([[1.0], [0.0]]))

    def test_minimum_fisher_regularisation_repeats_solver(self):
        solver = ConstantInversion(regularisation_selector=FixedSelector(1.0))
        mfr = MinimumFisherRegularisation(solver, mfi_num_max=2)
        derivatives = [sparse.eye_array(2, format='csc')]

        out, statistics = mfr(
            data=np.array([1.0, 2.0]),
            gmat=np.eye(2),
            derivatives=derivatives,
            errors=1.0,
        )

        np.testing.assert_allclose(out, np.ones(2))
        self.assertEqual(len(statistics), 2)
        self.assertTrue(all(np.isclose(entry['logalpha'], 0.0) for entry in statistics))

    def test_minimum_fisher_regularisation_validates_inversion_list_length(self):
        inversions = [ConstantInversion(regularisation_selector=FixedSelector(1.0))]

        self.assertRaises(
            ValueError,
            MinimumFisherRegularisation,
            inversions,
            mfi_num_max=2,
        )


class PackageExportTestCase(unittest.TestCase):
    def test_core_version_and_exports(self):
        self.assertTrue(tomotok.core.__version__)
        self.assertEqual(
            tomotok.core.__all__,
            ['geometry', 'inversions', 'regularisations'],
        )