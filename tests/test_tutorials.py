# Licensed under the EUPL-1.2 or later.
"""
Smoke tests that run the top-level tutorial scripts end to end.

These only check that a tutorial still runs without raising, using a headless
matplotlib backend. They do not assert on any numerical output, since the
tutorials are meant to be read and tweaked, not treated as behavioural specs.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / 'tutorials'


def run_tutorial(name, timeout=120):
    env = dict(os.environ, MPLBACKEND='Agg')
    result = subprocess.run(
        [sys.executable, str(TUTORIALS_DIR / name)],
        cwd=TUTORIALS_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            f'{name} exited with code {result.returncode}\n{result.stderr}'
        )


class TutorialSmokeTestCase(unittest.TestCase):
    def test_anisotropic_runs(self):
        run_tutorial('anisotropic.py')

    def test_nonnegative_runs(self):
        run_tutorial('nonnegative.py')

    def test_lame_fast_runs(self):
        run_tutorial('lame/fast.py')

    def test_ecpd2021_all_figures_runs(self):
        run_tutorial('ecpd2021/all_figures.py')


if __name__ == '__main__':
    unittest.main()
