# Licensed under the EUPL-1.2 or later.
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

from tomotok.core.geometry import RegularGrid
from tomotok.core.geometry.generators import calcam_sparse_line, calcam_sparse_line_3d, dense_line, sparse_line, sparse_line_3d
from tomotok.core.geometry.io import load_dense_gmat, load_sparse_gmat, save_dense_gmat, save_sparse_gmat


class RegularGridTestCase(unittest.TestCase):
    def setUp(self):
        self.grid = RegularGrid(2, 2, (1.0, 3.0), (-1.0, 1.0))

    def test_basic_properties(self):
        self.assertEqual(self.grid.size, 4)
        self.assertEqual(self.grid.shape, (2, 2))
        self.assertEqual(self.grid.extent, (1.0, 3.0, -1.0, 1.0))
        self.assertEqual(self.grid.centre, (2.0, 0.0))
        self.assertEqual(self.grid.vertices.shape, (9, 2))
        self.assertEqual(self.grid.faces.shape, (4, 4))
        np.testing.assert_array_equal(self.grid.faces[0], np.array([0, 1, 4, 3]))

    def test_volumes_and_corners(self):
        volumes = self.grid.volumes
        corners = self.grid.corners()
        mask = np.array([[True, False], [False, True]])
        masked_corners = self.grid.corners(mask=mask)

        self.assertEqual(volumes.shape, self.grid.shape)
        self.assertEqual(corners.shape, (2, 2, 4, 2))
        self.assertEqual(masked_corners.shape, (2, 4, 2))
        np.testing.assert_allclose(volumes[0], np.pi * np.array([3.0, 5.0]))

    def test_is_inside_methods(self):
        r = np.array([0.5, 3.5, 3.5, 0.5])
        z = np.array([-1.5, -1.5, 1.5, 1.5])

        center = self.grid.is_inside(r, z, method='center')
        any_corner = self.grid.is_inside(r, z, method='any')
        all_corners = self.grid.is_inside(r, z, method='all')

        self.assertTrue(np.all(center))
        self.assertTrue(np.all(any_corner))
        self.assertTrue(np.all(all_corners))
        self.assertRaises(ValueError, self.grid.is_inside, r, z, 'invalid')


class GeometryGeneratorsTestCase(unittest.TestCase):
    def setUp(self):
        self.grid = RegularGrid(2, 2, (1.0, 3.0), (-1.0, 1.0))
        self.starts = np.array([[1.5, 0.0, -0.8]])
        self.ends = np.array([[1.5, 0.0, 0.8]])
        self.pupil = np.array([0.0, 0.0, 0.0])
        self.endpoints = np.array([[[1.5, 0.0, 0.8]], [[1.5, 0.0, -0.8]]])

    def test_dense_and_sparse_line_match(self):
        dense = dense_line(self.starts, self.ends, self.grid, step=0.05)
        sparse_res = sparse_line(self.starts, self.ends, self.grid, step=0.05)

        self.assertEqual(dense.shape, (1, self.grid.size))
        self.assertEqual(sparse_res.shape, (1, self.grid.size))
        np.testing.assert_allclose(dense, sparse_res.toarray())

    def test_sparse_line_reshapes_higher_dimensional_input(self):
        starts = np.array([[[1.5, 0.0, -0.8]], [[1.5, 0.0, 0.0]]])
        ends = np.array([[[1.5, 0.0, 0.8]], [[1.5, 0.0, 0.8]]])

        dense = dense_line(starts, ends, self.grid, step=0.05)
        sparse_res = sparse_line(starts, ends, self.grid, step=0.05)

        self.assertEqual(dense.shape, (2, self.grid.size))
        self.assertEqual(sparse_res.shape, (2, self.grid.size))

    def test_line_generators_reject_invalid_shape(self):
        bad_starts = np.zeros((2, 2, 2))
        bad_ends = np.zeros((2, 2, 2))

        self.assertRaises(ValueError, dense_line, bad_starts, bad_ends, self.grid)
        self.assertRaises(ValueError, sparse_line, bad_starts, bad_ends, self.grid)

    def test_calcam_sparse_line_matches_sparse_line(self):
        starts = np.ones_like(self.endpoints) * self.pupil

        expected = sparse_line(starts, self.endpoints, self.grid, step=0.05)
        actual = calcam_sparse_line(self.pupil, self.endpoints, self.grid, step=0.05)

        np.testing.assert_allclose(actual.toarray(), expected.toarray())

    def test_deprecated_wrappers_work_and_warn(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            sparse_old = sparse_line_3d(
                rchord=np.array([[1.5, 1.5]]),
                vchord=np.array([[-0.8, 0.8]]),
                ychord=np.array([[0.0, 0.0]]),
                grid=self.grid,
                step=0.05,
            )
            calcam_old = calcam_sparse_line_3d(
                pupil=self.pupil,
                dirs=np.array([[[1.5, 0.0, 0.8]]]),
                grid=self.grid,
                step=0.05,
            )

        self.assertEqual(sparse_old.shape, (1, self.grid.size))
        self.assertEqual(calcam_old.shape, (1, self.grid.size))
        self.assertGreaterEqual(len(captured), 2)
        self.assertTrue(all(issubclass(item.category, DeprecationWarning) for item in captured))


class GeometryIoTestCase(unittest.TestCase):
    def setUp(self):
        self.grid = RegularGrid(2, 2, (1.0, 3.0), (-1.0, 1.0))

    def test_dense_roundtrip(self):
        gmat = np.arange(8, dtype=float).reshape(2, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'dense.h5'
            save_dense_gmat(path, gmat, self.grid)
            loaded, grid = load_dense_gmat(path)

        np.testing.assert_allclose(loaded, gmat)
        self.assertEqual(grid.shape, self.grid.shape)
        self.assertEqual(grid.rlims, self.grid.rlims)
        self.assertEqual(grid.zlims, self.grid.zlims)

    def test_sparse_roundtrip(self):
        gmat = sparse.csr_array(np.arange(8, dtype=float).reshape(2, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'sparse.h5'
            save_sparse_gmat(path, gmat, self.grid)
            loaded, grid = load_sparse_gmat(path)

        np.testing.assert_allclose(loaded.toarray(), gmat.toarray())
        self.assertEqual(grid.shape, self.grid.shape)
        self.assertEqual(grid.rlims, self.grid.rlims)
        self.assertEqual(grid.zlims, self.grid.zlims)

    def test_save_functions_validate_input_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = Path(tmpdir) / 'dense.h5'
            sparse_path = Path(tmpdir) / 'sparse.h5'

            self.assertRaises(ValueError, save_dense_gmat, dense_path, [[1, 2]], self.grid)
            self.assertRaises(ValueError, save_sparse_gmat, sparse_path, np.eye(4), self.grid)