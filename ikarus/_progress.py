"""Optional, dependency-free progress bars for sweeps and optimization.

Uses `tqdm <https://tqdm.github.io/>`_ when it is installed; otherwise falls back
to a minimal stderr bar, so ``progress=True`` always works.  ``tqdm`` is an
optional extra::

    pip install "ikarus-rcwa[progress]"

Public surface: :func:`progress` (wrap an iterable) and :func:`counter` (a manual
``.update()``/``.close()`` bar, used internally by the optimizer).
"""

from __future__ import annotations

import sys


def _have_tqdm() -> bool:
    try:
        import tqdm  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def progress(iterable, enable: bool = True, desc: str | None = None, total=None):
    """Wrap ``iterable`` in a progress bar; a no-op pass-through if ``enable`` is
    false.  ``total`` is inferred from ``len(iterable)`` when possible."""
    if not enable:
        return iterable
    if total is None:
        try:
            total = len(iterable)
        except TypeError:  # pragma: no cover - generators
            total = None
    if _have_tqdm():
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    return _fallback_iter(iterable, total, desc)


def counter(total, enable: bool = True, desc: str | None = None):
    """A manual progress counter exposing ``.update(n=1)`` and ``.close()``
    (also a context manager).  No-op when ``enable`` is false."""
    if not enable:
        return _NullCounter()
    if _have_tqdm():
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    return _FallbackCounter(total, desc)


class _NullCounter:
    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class _FallbackCounter:
    """A minimal stderr progress bar used when tqdm is not installed."""

    def __init__(self, total, desc=None):
        self.total = total
        self.n = 0
        self.desc = desc or "progress"
        self._draw()

    def update(self, n=1):
        self.n += n
        self._draw()

    def _draw(self):
        if self.total:
            frac = self.n / self.total
            filled = int(24 * frac)
            bar = "#" * filled + "-" * (24 - filled)
            sys.stderr.write(f"\r{self.desc}: [{bar}] {self.n}/{self.total} "
                             f"({100 * frac:3.0f}%)")
        else:
            sys.stderr.write(f"\r{self.desc}: {self.n}")
        sys.stderr.flush()

    def close(self):
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fallback_iter(iterable, total, desc):
    bar = _FallbackCounter(total, desc)
    try:
        for item in iterable:
            yield item
            bar.update(1)
    finally:
        bar.close()
