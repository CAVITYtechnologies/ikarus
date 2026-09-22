"""Pins the public API surface that 1.0 promises to keep.

``public_api.txt`` next to this file is the frozen list of exported symbols.
Adding one is a deliberate act: run ``python -m ikarus.tests.test_public_api``
to regenerate the file and commit the diff.  Removing or renaming one is a
breaking change and needs a major version, so a red test here is the point.

This is the executable half of the 1.0 stability promise; the prose half is
the deprecation policy in the docs.
"""

import importlib
import pathlib

SURFACE_FILE = pathlib.Path(__file__).with_name("public_api.txt")

# Every package whose ``__all__`` is part of the public promise.  ``ikarus.grad``
# is here too, but only materializes when the optional [grad] extra is installed
# (see _surface); ``ikarus.examples`` is demo code and deliberately unpinned.
MODULES = [
    "ikarus",
    "ikarus.core",
    "ikarus.shapes",
    "ikarus.inverse",
    "ikarus.materials",
    "ikarus.tools",
    "ikarus.visualization",
    "ikarus.grad",
]

# ikarus.grad's surface is empty unless JAX is installed, so it can only be
# pinned on machines that have the extra.
try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    _HAS_JAX = False


def _surface() -> list[str]:
    """Every exported symbol as ``module.name``, sorted."""
    out = []
    for mod_name in MODULES:
        if mod_name == "ikarus.grad" and not _HAS_JAX:
            continue
        mod = importlib.import_module(mod_name)
        exported = getattr(mod, "__all__", None)
        assert exported is not None, (
            f"{mod_name} has no __all__, so its public surface is undeclared. "
            f"Add one (see ikarus/materials/__init__.py for the pattern)."
        )
        for name in exported:
            assert hasattr(mod, name), (
                f"{mod_name}.{name} is listed in __all__ but does not exist -- "
                f"`from {mod_name} import {name}` would fail for users."
            )
            out.append(f"{mod_name}.{name}")
    return sorted(out)


def _expected() -> list[str]:
    lines = SURFACE_FILE.read_text().split()
    if not _HAS_JAX:
        lines = [x for x in lines if not x.startswith("ikarus.grad.")]
    return sorted(lines)


def test_public_surface_unchanged():
    actual, expected = _surface(), _expected()
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    assert not (added or removed), (
        "The public API surface changed.\n"
        f"  added:   {added or '(none)'}\n"
        f"  removed: {removed or '(none)'}\n"
        "Removing or renaming a symbol breaks the 1.0 promise and needs a major "
        "version. If the change is intended, regenerate the pin with:\n"
        "  python -m ikarus.tests.test_public_api"
    )


def test_no_duplicate_exports_within_a_module():
    """A repeated name in one __all__ is a copy-paste slip, not a decision."""
    for mod_name in MODULES:
        if mod_name == "ikarus.grad" and not _HAS_JAX:
            continue
        exported = getattr(importlib.import_module(mod_name), "__all__", [])
        dupes = {n for n in exported if exported.count(n) > 1}
        assert not dupes, f"{mod_name}.__all__ repeats {sorted(dupes)}"


if __name__ == "__main__":
    if not _HAS_JAX:
        raise SystemExit(
            "Refusing to regenerate without JAX installed: ikarus.grad's exports "
            'would be silently dropped from the pin. Install with:\n'
            '  pip install "ikarus-rcwa[grad]"'
        )
    SURFACE_FILE.write_text("\n".join(_surface()) + "\n")
    print(f"wrote {SURFACE_FILE} ({len(_surface())} symbols)")
