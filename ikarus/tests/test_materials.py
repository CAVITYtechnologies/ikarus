"""Tests for the material database and dispersion handling."""

import numpy as np
import pytest

from ikarus.core.materials import Material, MaterialLibrary, default_library


def test_constant_material():
    m = Material.constant(1.5 + 0.1j, name="x")
    assert np.isclose(m.index(500e-9), 1.5 + 0.1j)
    assert np.isclose(m.permittivity(500e-9), (1.5 + 0.1j) ** 2)


def test_builtin_materials_load():
    lib = default_library
    for name in ["Si", "SiO2", "TiO2", "Air", "Au", "GaN"]:
        assert name in lib.available()
        nk = lib.get(name, 600e-9)
        assert np.imag(nk) >= 0  # physics convention: k >= 0


def test_sio2_known_value():
    # Fused silica near 633 nm has n ~ 1.457.
    n = default_library.get("SiO2", 633e-9)
    assert abs(n.real - 1.457) < 0.01
    assert abs(n.imag) < 1e-6


def test_interpolation_is_smooth():
    lib = default_library
    wls = np.linspace(500e-9, 1500e-9, 50)
    n = np.array([lib.get("Si", w).real for w in wls])
    # No jumps larger than a reasonable dispersion slope.
    assert np.max(np.abs(np.diff(n))) < 0.5


def test_resolve_accepts_number_and_material():
    lib = MaterialLibrary()
    assert np.isclose(lib.get(2.0, 500e-9), 2.0)
    assert np.isclose(lib.get(Material.constant(1.3), 500e-9), 1.3)


def test_csv_import(tmp_path):
    csv = tmp_path / "mat.csv"
    csv.write_text("# wl_nm n k\n400 2.0 0.1\n800 1.8 0.05\n1200 1.7 0.0\n")
    lib = MaterialLibrary(tmp_path)
    mat = lib.add_from_file(csv, name="Custom")
    assert mat.name == "Custom"
    n = lib.get("Custom", 800e-9)
    assert abs(n.real - 1.8) < 1e-6
    assert abs(n.imag - 0.05) < 1e-6


def test_lorentz_model():
    lor = Material(name="L", lorentz={
        "eps_inf": 2.0,
        "oscillators": [{"f": 1.0, "w0": 4e15, "gamma": 1e14}],
    })
    eps = lor.permittivity(500e-9)
    assert np.imag(eps) > 0  # absorbing under exp(-i w t)


def test_tabulated_k_never_negative():
    # A cubic spline through tabulated k that decays to zero can undershoot below
    # zero between points -- unphysical gain under exp(-i w t); it must be clamped.
    lib = default_library
    wls = np.linspace(1200e-9, 2000e-9, 200)
    k = np.array([lib.get("aSi", w).imag for w in wls])
    assert (k >= 0).all(), f"negative k (gain) at {wls[k < 0] * 1e9} nm"


def test_synthetic_tabulated_k_clamped():
    m = Material(name="syn", n=np.full(4, 3.5),
                 wavelength_nm=np.array([1000.0, 1200.0, 1400.0, 1600.0]),
                 k=np.array([1e-3, 0.0, 0.0, 0.0]))
    wls = np.linspace(1000e-9, 1600e-9, 300)
    k = np.array([m.index(w).imag for w in wls])
    assert (k >= 0).all()


def test_csv_import_comma(tmp_path):
    # A genuine comma-separated file (with header) imports correctly, rather than
    # silently becoming an all-NaN Material.
    csv = tmp_path / "comma.csv"
    csv.write_text("lambda_nm,n,k\n400,2.0,0.1\n800,1.8,0.05\n1200,1.7,0.0\n")
    lib = MaterialLibrary(tmp_path)
    mat = lib.add_from_file(csv, name="Comma")
    assert not np.isnan(mat.n).any()
    n = lib.get("Comma", 800e-9)
    assert abs(n.real - 1.8) < 1e-6 and abs(n.imag - 0.05) < 1e-6


def test_csv_import_two_columns_no_k(tmp_path):
    csv = tmp_path / "nk.csv"
    csv.write_text("1064,3.66\n1550,3.48\n")
    lib = MaterialLibrary(tmp_path)
    mat = lib.add_from_file(csv, name="NK")
    assert not np.isnan(mat.n).any() and np.allclose(mat.k, 0.0)


def test_csv_import_raises_on_unparseable(tmp_path):
    # Unparseable data raises instead of silently poisoning the Material with NaN.
    bad = tmp_path / "bad.csv"
    bad.write_text("400,2.0,0.1\nfoo,bar,baz\n")
    lib = MaterialLibrary(tmp_path)
    with pytest.raises(ValueError):
        lib.add_from_file(bad, name="Bad")


def test_materials_subpackage_reexports():
    # `from ikarus.materials import default_library` works, matching `from ikarus`.
    from ikarus.materials import (Material as M2, MaterialLibrary as ML2,
                                  default_library as dl2)
    assert "Si" in dl2.available()
    assert M2 is Material and ML2 is MaterialLibrary
