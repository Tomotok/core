# Licensed under the EUPL-1.2 or later.
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'jax is not installed')
class JaxBackendTestCase(unittest.TestCase):
    def test_jax_engines_solve_linear_system(self):
        from tomotok.inversions.solvers.jax import JaxCholesky, JaxSolver

        a = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        expected = np.linalg.solve(a, b)

        np.testing.assert_allclose(JaxCholesky().solve(a, b), expected)
        np.testing.assert_allclose(JaxSolver().solve(a, b), expected)


@unittest.skipUnless(importlib.util.find_spec('sksparse') is not None, 'sksparse is not installed')
class SkSparseBackendTestCase(unittest.TestCase):
    def test_cholmod_engine_solves_linear_system(self):
        from tomotok.inversions.solvers.sksparse import SksparseCholesky

        a = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        expected = np.linalg.solve(a, b)

        np.testing.assert_allclose(SksparseCholesky().solve(a, b), expected)