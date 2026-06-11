# Tutorials

Task-oriented walkthroughs. Each is self-contained and runnable; together they
cover the everyday workflows. They assume you have read the
[Quick Start](../quickstart.md).

| Tutorial | You will learn |
|---|---|
| [Reflection & Transmission spectra](reflection-transmission.md) | Sweep wavelength, read totals and the specular order, handle absorption. |
| [Grating diffraction](gratings.md) | 1-D gratings, diffraction orders, exit angles, the grating equation. |
| [Metasurface simulation](metasurfaces.md) | Patterned 2-D layers from shapes, fields, phase. |
| [Parameter sweeps](parameter-sweeps.md) | Efficient sweeps, convergence studies, 2-D maps. |
| [Polarization analysis](polarization.md) | Linear angle sweeps, circular co/cross, chirality. |
| [Angular response](angular-response.md) | Incidence-angle sweeps, dispersion, Wood anomalies. |

All snippets use SI units (meters, degrees) and the physics \(\exp(-i\omega t)\)
convention. Most run in well under a second at the order counts shown.

!!! tip "Runnable example scripts"
    The package also ships full scripts under `ikarus/examples/` — see the
    [Examples Gallery](../examples-gallery.md). Run e.g.
    `python -m ikarus.examples.feature_tour`.
