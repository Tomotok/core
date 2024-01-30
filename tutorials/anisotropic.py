"""
This tutorial demonstrates how to check values of the anisotropic derivative matrix
"""
import matplotlib.pyplot as plt

from tomotok.core.derivative import anisotropic_derivative_matrix
from tomotok.core.geometry import RegularGrid
from tomotok.core.phantoms import elliptical_flux
from tomotok.core.tools.checkers import AnisotropicDerivativeChecker


grid = RegularGrid(50, 100, (0.5, 1), (-0.5, 0.5))


psi = elliptical_flux(grid.nr, grid.nz)

par = anisotropic_derivative_matrix(grid, psi, direction='parallel', scheme='forward')
per = anisotropic_derivative_matrix(grid, psi, direction='perpendicular', scheme='forward')

checker = AnisotropicDerivativeChecker(grid, psi, par, per)

plt.show()
