# Licensed under the EUPL-1.2 or later.
"""
Generates figures for the Biorthogonal Basis Decomposition documentation page.

The figures are not generated during the documentation build. Rerun this script and commit the figures
when a change of the code affects them; tests/test_docs_figures.py checks that the script still runs.

Uses a synthetic tangentially viewing camera with 200 x 200 pixels observing a grid of 50 x 50 nodes.
Computing the geometry matrix takes about a minute. An already computed matrix can be provided
using the ``--gmat`` option and a newly computed matrix can be stored using ``--save-gmat``.

Usage::

    python make_bob_figures.py [--gmat GMAT.h5] [--save-gmat GMAT.h5]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from tomotok.geometry import RegularGrid
from tomotok.geometry.generators import sparse_line
from tomotok.geometry.io import load_sparse_gmat, save_sparse_gmat
from tomotok.geometry.sightlines import generate_sightlines
from tomotok.inversions.bob import Bob
from tomotok.tools.phantoms import elliptical_flux, gaussian_on_flux

OUTPUT = Path(__file__).parent

# camera setup
PINHOLE = (0.8, 0.1, 0.2)
RESOLUTION = (200, 200)
FOV = (40, 40)
AXIS = (-1, 0.4, -0.3)

NOISE = 3  # standard deviation of noise added to synthetic image
THRESHOLD = 3.4  # thresholding sensitivity constant
SEED = 12345678
NODE = 1425  # node used for demonstration of basis vectors


def compute_gmat(
    resolution: tuple[int, int] = RESOLUTION, grid_shape: tuple[int, int] = (50, 50), step: float = 1e-4
) -> tuple[sparse.csr_array, RegularGrid]:
    """Computes geometry matrix of the synthetic camera with given resolution for a grid with shape (nr, nz)."""
    grid = RegularGrid(*grid_shape, rlims=(0.2, 0.7), zlims=(-0.25, 0.25))
    starts, ends = generate_sightlines(pinhole=PINHOLE, num=resolution, fov=FOV, axis=AXIS, length=3)
    gmat = sparse_line(starts, ends, grid, step=step, rmin=grid.rmin)
    return gmat, grid


def make_figures(
    gmat: sparse.csr_array,
    grid: RegularGrid,
    resolution: tuple[int, int] = RESOLUTION,
    output: Path = OUTPUT,
    node_index: int = NODE,
) -> None:
    """Inverts a synthetic image of a phantom and saves the figures to the output directory."""
    # phantom with hollow profile moved from the grid centre
    phantom = gaussian_on_flux(elliptical_flux(grid.nr, grid.nz), amplitude=2000, center=0.3, width=0.05)
    phantom = np.roll(phantom, (3, -5), (0, 1))

    # synthetic image with noise
    rng = np.random.default_rng(SEED)
    image = gmat @ phantom.flatten()
    image = image + rng.normal(0, NOISE, image.shape)

    # simple node basis, no de-noising capability
    basis = sparse.eye_array(grid.size, format='csc')
    bob = Bob()
    bob.decompose(gmat, basis, reg_factor=1e-10)

    reconstruction = bob(image).reshape(grid.shape)
    thresholded = bob.thresholding(image, c=THRESHOLD).reshape(grid.shape)

    # basis vectors for a single node
    node = np.zeros(grid.size)
    node[node_index] = 1
    projection = gmat @ node
    adjoint = bob.adjoint_basis[:, [node_index]].toarray().flatten()

    plt.rcParams['image.cmap'] = 'cividis'

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.4), facecolor='white')
    axs[0].pcolormesh(*grid.border_mesh, node.reshape(grid.shape), rasterized=True)
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('R [m]')
    axs[0].set_ylabel('z [m]')
    axs[0].set_title(r'$\mathbf{b}_i$')
    axs[1].imshow(projection.reshape(resolution))
    axs[1].set_title(r'$\mathbf{a}_i$')
    axs[2].imshow(adjoint.reshape(resolution))
    axs[2].set_title(r'$\tilde{\mathbf{b}}_i$')
    for ax in axs[1:]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / 'bob_basis.png', dpi=150)
    plt.close(fig)

    limits = dict(
        vmin=min(phantom.min(), reconstruction.min(), thresholded.min()),
        vmax=max(phantom.max(), reconstruction.max(), thresholded.max()),
    )
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True, facecolor='white', layout='constrained')
    for ax, values, title in zip(axs, (phantom, reconstruction, thresholded), ('Phantom', 'Reconstructed', 'Thresholded')):
        cmesh = ax.pcolormesh(*grid.border_mesh, values, rasterized=True, **limits)
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_title(title)
    axs[0].set_ylabel('z [m]')
    fig.colorbar(cmesh, ax=axs, label='Emissivity [-]', shrink=0.8)
    fig.savefig(output / 'bob_thresholding.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gmat', type=Path, help='load geometry matrix from file instead of computing it')
    parser.add_argument('--save-gmat', type=Path, help='save computed geometry matrix to file')
    args = parser.parse_args()

    if args.gmat is not None:
        gmat, grid = load_sparse_gmat(args.gmat)
    else:
        gmat, grid = compute_gmat()
        if args.save_gmat is not None:
            save_sparse_gmat(args.save_gmat, gmat, grid)
    make_figures(gmat, grid)


if __name__ == '__main__':
    main()
