# Licensed under the EUPL-1.2 or later.
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from tomotok.geometry import RegularGrid, sightlines
from tomotok.tools import Diagnostic, Tokamak
from tomotok.tools import __all__ as tools_all
from tomotok.tools import hdf, phantoms
from tomotok.tools.containers import Magnetics


class ToolsExportsTestCase(unittest.TestCase):
    def test_public_exports_are_available(self):
        self.assertIn("Diagnostic", tools_all)
        self.assertIn("Tokamak", tools_all)

        self.assertTrue(callable(Diagnostic))
        self.assertTrue(callable(Tokamak))


class IoTemplatesTestCase(unittest.TestCase):
    def test_diagnostic_template_requires_override(self):
        diag = Diagnostic()
        self.assertRaises(NotImplementedError, diag.load_data, 12345)

    def test_tokamak_template_requires_override(self):
        tokamak = Tokamak()
        self.assertRaises(NotImplementedError, tokamak.load_magnetic_field, 12345)

    def test_tokamak_validates_divertor_definition(self):
        self.assertRaises(ValueError, Tokamak, divertor_area_r=[1.0], divertor_area_z=None)
        self.assertRaises(ValueError, Tokamak, divertor_area_r=[1.0, 2.0], divertor_area_z=[0.0])

        tokamak = Tokamak(divertor_area_r=[1.0, 1.5], divertor_area_z=[0.0, -0.2])
        np.testing.assert_allclose(tokamak.divertor_area_r, np.array([1.0, 1.5]))
        np.testing.assert_allclose(tokamak.divertor_area_z, np.array([0.0, -0.2]))


class HdfSparseRoundtripTestCase(unittest.TestCase):
    def test_sparse_roundtrip_for_supported_formats(self):
        matrices = [
            sparse.csr_array(np.array([[1.0, 0.0], [2.0, 3.0]])),
            sparse.csc_array(np.array([[0.0, 4.0], [5.0, 0.0]])),
            sparse.dia_array(np.diag([7.0, 8.0])),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrices.h5"
            with h5py.File(path, "w") as fl:
                for i, matrix in enumerate(matrices):
                    grp = fl.create_group(f"m{i}")
                    hdf.sparse_to_hdf(matrix, grp)

            with h5py.File(path, "r") as fl:
                loaded = [hdf.hdf_to_sparse(fl[f"m{i}"]) for i in range(len(matrices))]

        for expected, actual in zip(matrices, loaded):
            np.testing.assert_allclose(actual.toarray(), expected.toarray())

    def test_sparse_to_hdf_rejects_unsupported_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.h5"
            with h5py.File(path, "w") as fl:
                grp = fl.create_group("matrix")
                self.assertRaises(TypeError, hdf.sparse_to_hdf, np.eye(2), grp)


class ContainersTestCase(unittest.TestCase):
    def setUp(self):
        self.time_axis = np.array([0.0, 1.0, 2.0], dtype=float)
        self.radial = np.linspace(1.0, 2.0, 5)
        self.vertical = np.linspace(-1.0, 1.0, 4)

        rr, zz = np.meshgrid(self.radial, self.vertical)
        flux = np.empty((self.time_axis.size, self.vertical.size, self.radial.size), dtype=float)
        for i, time in enumerate(self.time_axis):
            flux[i] = rr + zz + time

        self.magnetics = Magnetics(
            time_axis=self.time_axis,
            vertical=self.vertical,
            radial=self.radial,
            flux=flux,
        )

        self.grid = RegularGrid(nr=3, nz=4, rlims=(1.0, 2.0), zlims=(-1.0, 1.0))

    def test_interpolate_keeps_duplicates_when_unique_false(self):
        tvec = np.array([0.2, 0.2, 1.4])

        interpolated = self.magnetics.interpolate(self.grid, tvec=tvec, unique=False)

        self.assertEqual(interpolated.time_axis.size, 3)
        self.assertEqual(interpolated.flux.shape, (3, self.grid.nz, self.grid.nr))

    def test_interpolate_deduplicates_when_unique_true(self):
        tvec = np.array([0.2, 0.2, 1.4])

        interpolated = self.magnetics.interpolate(self.grid, tvec=tvec, unique=True)

        np.testing.assert_allclose(interpolated.time_axis, np.array([1.0, 2.0]))
        self.assertEqual(interpolated.flux.shape, (2, self.grid.nz, self.grid.nr))
        np.testing.assert_allclose(interpolated.radial, self.grid.r_center)
        np.testing.assert_allclose(interpolated.vertical, self.grid.z_center)


class PhantomsTestCase(unittest.TestCase):
    def test_elliptical_flux_rejects_invalid_span_tuple(self):
        self.assertRaises(ValueError, phantoms.elliptical_flux, 4, 4, span=(1.0, 2.0, 3.0))

    def test_deprecated_iso_psi_and_gauss_iso_warn(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            iso = phantoms.iso_psi(4, 5, span=1.2)
            gauss_iso = phantoms.gauss_iso(4, 5)

        self.assertEqual(iso.shape, (5, 4))
        self.assertEqual(gauss_iso.shape, (5, 4))
        self.assertGreaterEqual(len(captured), 2)
        self.assertTrue(all(issubclass(item.category, DeprecationWarning) for item in captured[:2]))


class SightlinesTestCase(unittest.TestCase):
    def test_generate_directions_with_array_axis(self):
        dirs = sightlines.generate_directions(
            num=(2, 2),
            fov=(20.0, 10.0),
            axis=np.array([1.0, 0.0, 0.0]),
            length=2.0,
        )

        self.assertEqual(dirs.shape, (4, 3))
        norms = np.linalg.norm(dirs, axis=1)
        self.assertTrue(np.all(norms >= 2.0))

    def test_save_los_and_validate_detector_names(self):
        starts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ends = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "los.json"
            sightlines.save_los(path, starts, ends, detector_names="camera_a")

            with path.open("r") as fl:
                data = json.load(fl)

        self.assertIn("camera_a", data)
        self.assertEqual(len(data["camera_a"]["startpoints"]), 2)
        self.assertEqual(len(data["camera_a"]["endpoints"]), 2)

    def test_save_los_rejects_shape_mismatch(self):
        starts = np.array([[0.0, 0.0, 0.0]])
        ends = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        self.assertRaises(ValueError, sightlines.save_los, "dummy.json", starts, ends)
