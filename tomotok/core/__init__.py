# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences.
# Licensed under the EUPL-1.2 or later.
"""
Legacy compatibility layer.

This package is deprecated and kept only so that old imports of the form
``tomotok.core.inversions``, ``tomotok.core.geometry`` and
``tomotok.core.regularisations`` keep working after the namespace
reorganisation that dropped the ``core`` prefix. It re-exports
:mod:`tomotok.geometry`, :mod:`tomotok.inversions` and
:mod:`tomotok.regularisations` and emits a :class:`DeprecationWarning` on
import. New code should import those packages directly instead of
:mod:`tomotok.core`.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from warnings import warn

from .. import geometry
from .. import inversions
from .. import regularisations


warn(
    "tomotok.core is deprecated and kept for compatibility. "
    "Use tomotok.geometry, tomotok.inversions and tomotok.regularisations directly.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    "geometry",
    "inversions",
    "regularisations",
]

try:
    __version__ = version("tomotok")
except PackageNotFoundError:
    # Local fallback for editable/dev trees before package metadata is available.
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    if version_path.exists():
        __version__ = version_path.read_text(encoding="utf-8").strip()
    else:
        __version__ = "0+unknown"
