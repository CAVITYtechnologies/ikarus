"""Tests for anisotropic (tensor permittivity) materials.

Scope: in-plane tensor + distinct z response, i.e. ``eps = [[exx, exy, 0],
[eyx, eyy, 0], [0, 0, ezz]]`` -- wave plates at any in-plane orientation,
c-plates, and patterned birefringent structures.  Validated against exact
analytic equivalences (below) and against FMMax ``eigensolve_anisotropic_media``
(see ``ikarus/core/_normalvector.py`` and the dev validation scripts).

Key exact identities used here (all at normal incidence unless said otherwise):

* ``linear_pol_angle=0`` is TE = E along **y**; ``90`` is E along **x**.
* A uniform a-plate ``(n_x, n_y, n_z)`` seen by x-polarized light behaves
  exactly like an isotropic slab of index ``n_x`` (and y-pol like ``n_y``).
* A c-plate ``(n_o, n_o, n_e)`` is exactly isotropic at normal incidence and
  for oblique TE (no E_z); only oblique TM feels ``eps_zz``.
"""

import numpy as np
import pytest

from ikarus import RCWA, AnisotropicMaterial, shapes, uniaxial

WL = 1550e-9


def _slab(material, pol_angle=0, h=700e-9, theta=0, n_orders=(2, 2)):
    rc = RCWA(period_x=1e-6, period_y=1e-6, n_orders=n_orders)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_uniform_layer(h, material)
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=WL, theta=theta, polarization="linear",
                  linear_pol_angle=pol_angle)
    return rc.simulate()[2]


def test_isotropic_tuple_collapses_to_scalar():
    """(n, n, n) must reproduce the plain scalar material *exactly* -- it is
    detected and routed through the identical isotropic code path."""
    r_scalar = _slab(1.9)
    r_tuple = _slab((1.9, 1.9, 1.9))
    assert r_scalar.R_total == r_tuple.R_total
    assert r_scalar.T_phase == r_tuple.T_phase


def test_aplate_each_axis_matches_isotropic_slab():
    """x-polarized light through (n_x, n_y, n_z) == isotropic slab of n_x, and
    y-polarized == isotropic slab of n_y, exactly (pol_angle=90 is E||x)."""
    aniso = (1.9, 2.4, 2.4)
    assert _slab(aniso, pol_angle=90).R_total == pytest.approx(
        _slab(1.9, pol_angle=90).R_total, abs=1e-12)
    assert _slab(aniso, pol_angle=0).R_total == pytest.approx(
        _slab(2.4, pol_angle=0).R_total, abs=1e-12)
    # and the two axes genuinely differ (the layer is birefringent)
    assert abs(_slab(aniso, 0).R_total - _slab(aniso, 90).R_total) > 0.01


def test_waveplate_retardance_matches_isotropic_pair():
    """The transmission-phase difference between the two axes equals the
    difference between the corresponding isotropic slabs, exactly."""
    h = 2e-6
    aniso = (1.9, 2.4, 2.4)
    ret_aniso = (_slab(aniso, 90, h).T_phase - _slab(aniso, 0, h).T_phase)
    ret_iso = (_slab(1.9, 90, h).T_phase - _slab(2.4, 0, h).T_phase)
    assert ret_aniso == pytest.approx(ret_iso, abs=1e-12)


def test_cplate_zz_selectivity():
    """eps_zz is invisible at normal incidence and to oblique TE (no E_z), but
    must change oblique TM -- the sharpest test of the longitudinal path."""
    cplate = (1.9, 1.9, 2.4)
    # normal incidence: exactly the isotropic n_o slab
    assert _slab(cplate, 90).R_total == pytest.approx(
        _slab(1.9, 90).R_total, abs=1e-12)
    # oblique TE: still exactly isotropic
    assert _slab(cplate, 0, theta=40).R_total == pytest.approx(
        _slab(1.9, 0, theta=40).R_total, abs=1e-10)
    # oblique TM: eps_zz active
    assert abs(_slab(cplate, 90, theta=40).R_total
               - _slab(1.9, 90, theta=40).R_total) > 1e-3


def test_rotated_plate_symmetry_and_offdiagonal():
    """A 45-degree rotated uniaxial plate treats x- and y-pol identically (exact
    mirror symmetry), and differs from the unrotated plate -- this exercises the
    off-diagonal eps_xy path end to end."""
    plate45 = uniaxial(1.9, 2.4, axis=45.0)
    r_x = _slab(plate45, 90)
    r_y = _slab(plate45, 0)
    assert r_x.R_total == pytest.approx(r_y.R_total, abs=1e-12)
    assert abs(r_x.energy_balance - 1.0) < 1e-7
    # the rotation genuinely does something
    assert abs(r_x.R_total - _slab((2.4, 1.9, 1.9), 90).R_total) > 1e-3


def test_uniaxial_helper_axes():
    m = uniaxial(1.5, 1.7, axis="z")
    assert (m.n_x, m.n_y, m.n_z, m.angle) == (1.5, 1.5, 1.7, 0.0)
    m = uniaxial(1.5, 1.7, axis="x")
    assert (m.n_x, m.n_y, m.n_z) == (1.7, 1.5, 1.5)
    m = uniaxial(1.5, 1.7, axis="y")
    assert (m.n_x, m.n_y, m.n_z) == (1.5, 1.7, 1.5)
    m = uniaxial(1.5, 1.7, axis=30.0)
    assert (m.n_x, m.n_y, m.n_z, m.angle) == (1.7, 1.5, 1.5, 30.0)
    with pytest.raises(ValueError, match="axis"):
        uniaxial(1.5, 1.7, axis="q")


def test_anisotropic_tensor_components_rotation():
    """The in-plane rotation produces the textbook tensor components."""
    from ikarus import default_library
    m = AnisotropicMaterial(2.0, 3.0, 1.5, angle=30.0)
    exx, exy, eyx, eyy, ezz = m.permittivity_tensor(WL, default_library)
    e1, e2 = 4.0, 9.0
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    assert exx == pytest.approx(e1 * c * c + e2 * s * s)
    assert eyy == pytest.approx(e1 * s * s + e2 * c * c)
    assert exy == pytest.approx((e1 - e2) * s * c)
    assert eyx == exy                      # reciprocal (non-magneto-optic)
    assert ezz == pytest.approx(2.25)


def test_patterned_anisotropic_energy_conserved():
    """A lossless birefringent pillar conserves energy under every rule."""
    mask = shapes.circle(center=(0.5, 0.5), radius=0.3, grid_shape=(64, 64))
    for fac in ("laurent", "li", "auto"):
        rc = RCWA(period_x=800e-9, period_y=800e-9, resolution=(64, 64),
                  n_orders=(6, 6), factorization=fac)
        rc.add_uniform_layer(np.inf, "Air")
        rc.add_layer(500e-9, mask.astype(int), ["Air", (2.0, 2.3, 2.2)])
        rc.add_uniform_layer(np.inf, "SiO2")
        rc.set_source(wavelength=WL, theta=0, polarization="linear",
                      linear_pol_angle=30)
        res = rc.simulate()[2]
        assert abs(res.energy_balance - 1.0) < 1e-4, fac


def test_patterned_anisotropic_pinned_to_fmmax():
    """Absolute regression guard: a birefringent pillar validated against FMMax
    ``eigensolve_anisotropic_media`` (NORMAL formulation).  Ikarus's converged
    value is R ~= 0.0253 (stable 0.02542/0.02537/0.02526 at M=8/12/16), with
    FMMax's own converged value 0.0249 (it wobbles with truncation; terminal
    agreement ~3e-4).  E||y here means linear_pol_angle=0."""
    mask = shapes.circle(center=(0.5, 0.5), radius=0.3, grid_shape=(96, 96))
    rc = RCWA(period_x=800e-9, period_y=800e-9, resolution=(96, 96),
              n_orders=(8, 8), factorization="auto")
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(500e-9, mask.astype(int), ["Air", (2.0, 2.3, 2.2)])
    rc.add_uniform_layer(np.inf, 1.4440)
    rc.set_source(wavelength=WL, theta=0, polarization="linear",
                  linear_pol_angle=0)
    assert rc.simulate()[2].R_total == pytest.approx(0.0254, abs=0.002)


def test_anisotropic_cover_rejected():
    rc = RCWA(period_x=1e-6, period_y=1e-6, n_orders=(2, 2))
    rc.add_uniform_layer(np.inf, (1.5, 1.6, 1.7))
    rc.add_uniform_layer(200e-9, "Si")
    rc.add_uniform_layer(np.inf, "SiO2")
    rc.set_source(wavelength=WL, theta=0, polarization="linear")
    with pytest.raises(ValueError, match="isotropic"):
        rc.simulate()
