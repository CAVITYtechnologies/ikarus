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
