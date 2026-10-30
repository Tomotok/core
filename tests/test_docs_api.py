# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences.
# Licensed under the EUPL-1.2 or later.
"""Checks that the public API is listed in the API reference pages of the documentation."""
import importlib
import inspect
import re
import unittest
from pathlib import Path

API = Path(__file__).parents[1] / 'docs' / 'source' / 'api'
PACKAGES = (
    'tomotok.geometry',
    'tomotok.regularisations',
    'tomotok.inversions',
    'tomotok.inversions.solvers',
    'tomotok.tools',
)


def documented_names() -> set[str]:
    """Returns fully qualified names of objects listed in autosummary directives of the API pages."""
    names = set()
    for page in API.glob('*.rst'):
        module = None
        in_autosummary = False
        for line in page.read_text().splitlines():
            directive = re.match(r'\.\. (?:currentmodule|module):: (\S+)', line)
            if directive:
                module = directive.group(1)
                in_autosummary = False
            elif line.startswith('.. autosummary::'):
                in_autosummary = True
            elif in_autosummary and re.match(r'    [A-Za-z_][\w.]*$', line):
                names.add(f'{module}.{line.strip()}')
            elif in_autosummary and line and not line.startswith(' '):
                in_autosummary = False
    return names


class MissingDependency(Exception):
    """An optional dependency required to import a documented object is not installed."""


def resolve(name: str):
    """Imports the object with given fully qualified name."""
    parts = name.split('.')
    obj = None
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ModuleNotFoundError as err:
            if err.name is not None and not err.name.startswith('tomotok'):
                raise MissingDependency(err.name) from err
            continue
        break
    for part in parts[i:]:
        obj = getattr(obj, part)
    return obj


class ApiReferenceTestCase(unittest.TestCase):
    def test_public_api_is_documented(self):
        documented = set()
        for name in documented_names():
            try:
                documented.add(id(resolve(name)))
            except MissingDependency:
                continue
        for package in PACKAGES:
            module = importlib.import_module(package)
            for name in module.__all__:
                obj = getattr(module, name)
                if inspect.ismodule(obj):
                    continue
                with self.subTest(object=f'{package}.{name}'):
                    self.assertIn(id(obj), documented, f'{package}.{name} is not listed in docs/source/api')
