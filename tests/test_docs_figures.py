# Licensed under the EUPL-1.2 or later.
"""Checks that the scripts generating figures for the documentation still run with the current code."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

FIGURES = Path(__file__).parents[1] / 'docs' / 'source' / 'methods' / 'figures'


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, FIGURES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BobFiguresTestCase(unittest.TestCase):
    def test_bob_figures_script_runs_on_small_configuration(self):
        script = load_script('make_bob_figures')
        resolution = (30, 30)
        gmat, grid = script.compute_gmat(resolution=resolution, grid_shape=(10, 10), step=1e-3)
        node_index = grid.size // 2 + grid.nr // 2

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            script.make_figures(gmat, grid, resolution=resolution, output=output, node_index=node_index)

            for name in ('bob_basis.png', 'bob_thresholding.png'):
                self.assertGreater((output / name).stat().st_size, 0)
