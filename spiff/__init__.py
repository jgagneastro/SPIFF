"""Local import shim for the src-layout SPIFF package."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "spiff"
if not _SRC_PACKAGE_DIR.is_dir():
    raise ImportError(f"Cannot find SPIFF source package at {_SRC_PACKAGE_DIR}")

# Resolve submodules like `spiff.examples` from `src/spiff` when running Python
# directly from the repository root without an editable install.
__path__ = [str(_SRC_PACKAGE_DIR)]

_src_init = _SRC_PACKAGE_DIR / "__init__.py"
exec(compile(_src_init.read_text(encoding="utf-8"), str(_src_init), "exec"), globals(), globals())
