"""MkDocs build hook: expose the package version to templates.

Reads ``version`` from ``pyproject.toml`` and sets ``config.extra.version`` so the
announcement banner (and anywhere else) always shows the current release without a
manual edit. Wired in via ``hooks:`` in ``mkdocs.yml``.
"""

import re
from pathlib import Path


def on_config(config, **kwargs):
    pyproject = Path(__file__).resolve().parent / "pyproject.toml"
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
    extra = config.get("extra") or {}
    extra["version"] = match.group(1) if match else ""
    config["extra"] = extra
    return config
