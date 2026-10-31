"""Tests for the material database and dispersion handling."""

import warnings

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


# -- shape-preserving interpolation ---------------------------------------
# A natural cubic spline through a coarse, strongly curved n/k table overshoots
# between samples.  On the shipped Si table that returned k(650 nm) = 0.0079 --
# below *both* bracketing samples (0.024 at 600 nm, 0.011 at 700 nm) -- and so
# understated silicon's absorption by a factor of two.  These pin the fix.

def test_interpolation_never_leaves_the_sample_bracket():
    for name in default_library.available():
        mat = default_library.resolve(name)
        if mat.wavelength_nm is None or mat.wavelength_nm.size < 3:
            continue
        wl, n, k = mat.wavelength_nm, mat.n, mat.k
        for i in range(wl.size - 1):
            probe = np.linspace(wl[i], wl[i + 1], 11)
            got = mat.index(probe * 1e-9)
            for values, part in ((n, got.real), (k, got.imag)):
                lo, hi = min(values[i], values[i + 1]), max(values[i], values[i + 1])
                span = max(hi - lo, 1e-12)
                assert part.min() >= lo - 1e-9 * span, f"{name}: undershoot in [{wl[i]}, {wl[i+1]}] nm"
                assert part.max() <= hi + 1e-9 * span, f"{name}: overshoot in [{wl[i]}, {wl[i+1]}] nm"


def test_silicon_absorption_at_650nm_is_not_halved():
    # refractiveindex.info (main/Si/Aspnes) gives k = 0.0164 here.  The coarse
    # shipped table cannot match that exactly, but it must land in the right
    # ballpark rather than the old 0.0079.
    k = default_library.get("Si", 650e-9).imag
    assert 0.012 < k < 0.020, f"Si k(650 nm) = {k}, expected ~0.016"


# -- structured provenance ------------------------------------------------

def test_material_source_survives_a_save_round_trip(tmp_path):
    lib = MaterialLibrary(tmp_path)
    src = {"database": "refractiveindex.info", "shelf": "main", "book": "X", "page": "Y"}
    lib.save(Material(name="X", wavelength_nm=np.array([400.0, 500.0, 600.0]),
                      n=np.array([2.0, 1.9, 1.8]), k=np.zeros(3), source=src))
    assert MaterialLibrary(tmp_path).resolve("X").source == src


def test_flat_k_table_interpolates_without_warning():
    # A transparent material's k is 0 over the whole band.  PCHIP's slope
    # limiter divides by the secant slopes, so a flat run warns unless guarded.
    mat = Material(name="clear", wavelength_nm=np.linspace(300.0, 2000.0, 40),
                   n=np.linspace(2.0, 1.9, 40), k=np.zeros(40))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = mat.index(np.linspace(300e-9, 2000e-9, 97))
    assert np.all(got.imag == 0.0)
    assert np.isfinite(got).all()


# -- provenance of the shipped database -----------------------------------
# House rule: optical constants are cited, never remembered.  Before these,
# most shipped tables were coarse and marked "representative" -- i.e. written
# from memory -- and that is how Si ended up ~2x off in k.  scripts/
# regenerate_materials.py rebuilds them; this stops one drifting back.

# Air is n = 1 exactly, by the convention that indices are relative to vacuum,
# so it is the one entry with nothing to cite.
_UNCITED = {"Air"}


def test_every_shipped_material_cites_its_source():
    for name in default_library.available():
        if name in _UNCITED:
            continue
        src = default_library.resolve(name).source
        assert src, f"{name} carries no provenance for its optical constants"
        entries = src["entries"]
        assert entries, f"{name}: empty provenance"
        for e in entries:
            assert e["database"] == "refractiveindex.info"
            assert e["shelf"] and e["book"] and e["page"]
            assert e["reference"], f"{name}: {e['page']} carries no literature reference"
        assert src["rationale"], f"{name}: no stated reason for choosing this entry"


def test_no_shipped_material_claims_to_be_representative():
    for name in default_library.available():
        if name in _UNCITED:
            continue
        comment = default_library.resolve(name).comment.lower()
        assert "representative" not in comment, (
            f"{name}: 'representative' means the numbers were written from memory"
        )


def test_shipped_materials_cover_the_band_they_are_used_at():
    # Interpolation clamps to the end samples instead of refusing, so a table
    # that stops short turns into a silently wrong answer rather than an error.
    expected = {
        "Si": (300, 2000), "aSi": (300, 2000), "SiO2": (300, 2000),
        "Si3N4": (310, 2000), "TiO2": (300, 2000), "GaP": (300, 2000),
        "GaN": (350, 2000), "Au": (300, 1937), "Ag": (188, 1937),
    }
    for name, (lo, hi) in expected.items():
        wl = default_library.resolve(name).wavelength_nm
        assert wl[0] <= lo + 1e-6 and wl[-1] >= hi - 1e-6, (
            f"{name} covers {wl[0]:.0f}-{wl[-1]:.0f} nm, expected {lo}-{hi} nm"
        )


def test_telecom_and_visible_indices_are_sane():
    # Spot values an optics person would recognise; a regenerated database that
    # fails these picked the wrong entry (e.g. amorphous data for "Si").
    lib = default_library
    assert abs(lib.get("SiO2", 1550e-9).real - 1.4440) < 1e-3     # fused silica
    assert abs(lib.get("Si3N4", 1550e-9).real - 1.996) < 5e-3     # stoichiometric
    assert abs(lib.get("Si", 1550e-9).real - 3.476) < 5e-3        # crystalline
    assert lib.get("aSi", 1550e-9).real > lib.get("Si", 1550e-9).real   # a-Si is higher
    assert lib.get("Ag", 1550e-9).imag > 10                        # good metal
    assert lib.get("Au", 1550e-9).imag > 10
