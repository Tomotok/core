# Licensed under the EUPL-1.2 or later.
import unittest

from tomotok.core.derivative import derivative_matrix, anisotropic_derivative_matrix
from tomotok.core.geometry import RegularGrid
from tomotok.core.phantoms import elliptical_flux


class IsotropicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = RegularGrid(100, 100, (1, 3), (-1, 1))
        return

    def test_direction(self):
        directions = ['left', 'bottom-left', 'bottom', 'bottom-right', 'right', 'top-right', 'top', 'top-left']
        for direction in directions:
            derivative_matrix(self.grid, direction)
        self.assertRaises(ValueError, derivative_matrix, self.grid, 'wrong')

    def test_scheme(self):
        schemes = ['central', 'forward', 'backward', 'second']
        for scheme in schemes:
            dm = derivative_matrix(self.grid, 'right', scheme)
            self.assertEqual(dm.shape, (self.grid.size, self.grid.size))
            # TODO: test for value checking in a selected row?
            # idx = self.grid.nz // 2 * self.grid.nr + self.grid.nr // 2
            # sel = dm[idx]
        self.assertRaises(ValueError, derivative_matrix, self.grid, 'left', 'wrong')


class AnisotropicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        nr = 100
        nz = 100
        self.grid = RegularGrid(nr, nz, (1, 3), (-1, 1))
        self.flux = elliptical_flux(nr, nz)
        return

    def test_direction(self):
        directions = ['parallel', 'perpendicular']
        for direction in directions:
            anisotropic_derivative_matrix(self.grid, self.flux, direction)
        self.assertRaises(ValueError, anisotropic_derivative_matrix, self.grid, self.flux, 'wrong')

    def test_scheme(self):
        schemes = ['forward', 'backward']
        for scheme in schemes:
            dm = anisotropic_derivative_matrix(self.grid, self.flux, scheme=scheme)
            self.assertEqual(dm.shape, (self.grid.size, self.grid.size))
        self.assertRaises(ValueError, anisotropic_derivative_matrix, self.grid, self.flux, scheme='wrong')
