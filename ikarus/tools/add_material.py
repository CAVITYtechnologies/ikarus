"""CLI to import a custom material from tabulated ``n, k`` data.

Usage
-----
    python -m ikarus.tools.add_material my_material.csv --name MyMaterial

The input is a whitespace- or comma-delimited text/CSV file with columns
``wavelength_nm  n  [k]`` (``k`` optional, defaults to 0; ``#`` lines ignored).
The data is sorted, stored as JSON in the package material database and is then
available by name to :class:`ikarus.RCWA`.
"""

from __future__ import annotations

import argparse
import sys

from ..core.materials import MaterialLibrary, _material_from_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ikarus.tools.add_material",
        description="Import tabulated n,k data into the Ikarus material database.",
    )
    parser.add_argument("path", help="CSV/text file: columns wavelength_nm n [k]")
    parser.add_argument("--name", required=True, help="Name to store the material under")
    parser.add_argument("--comment", default="", help="Optional description")
    parser.add_argument("--db", default=None, help="Target database directory")
    args = parser.parse_args(argv)

    lib = MaterialLibrary(args.db) if args.db else MaterialLibrary()
    mat = _material_from_csv(__import__("pathlib").Path(args.path), args.name)
    mat.name = args.name
    if args.comment:
        mat.comment = args.comment
    out = lib.save(mat)

    n_pts = len(mat.wavelength_nm)
    lam0, lam1 = mat.wavelength_nm[0], mat.wavelength_nm[-1]
    print(f"Stored '{args.name}' ({n_pts} points, {lam0:.0f}-{lam1:.0f} nm) -> {out}")
    print(f"Use it via RCWA(...).add_uniform_layer(height, material='{args.name}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
