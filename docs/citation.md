# Citation

If you use Ikarus in academic work, please cite the software.

## How to cite

> Shelling Neto, L. *Ikarus: high-precision 2-D RCWA simulation for periodic
> photonic structures* (version 0.2.0). CAVITY technologies UG, 2026.
> https://github.com/CAVITYtechnologies/ikarus

## BibTeX

```bibtex
@software{ikarus_rcwa,
  author  = {Shelling Neto, Liam},
  title   = {{Ikarus}: High-Precision 2-D RCWA Simulation for Periodic
             Photonic Structures},
  year    = {2026},
  version = {0.2.0},
  publisher = {CAVITY technologies UG},
  url     = {https://github.com/CAVITYtechnologies/ikarus}
}
```

Replace `version` with the release you actually used (`import ikarus;
ikarus.__version__`, or the installed `ikarus-rcwa` distribution version).

!!! tip "Add a DOI for archival citation"
    For a permanently citable, versioned reference, archive a release on
    [Zenodo](https://zenodo.org/) (it integrates with GitHub releases) and add the
    resulting DOI to the `doi`/`url` field above.

## Background references

Ikarus implements the scattering-matrix formulation of rigorous coupled-wave
analysis (the Fourier Modal Method). The method rests on:

- M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating
  diffraction," *J. Opt. Soc. Am.* **71**, 811 (1981).
- M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, "Formulation for stable
  and efficient implementation of the rigorous coupled-wave analysis of binary
  gratings," *J. Opt. Soc. Am. A* **12**, 1068 (1995).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic
  structures," *J. Opt. Soc. Am. A* **13**, 1870 (1996) — Fourier factorization
  rules (the inverse rule Ikarus does not yet use).
- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling
  layered diffraction gratings," *J. Opt. Soc. Am. A* **13**, 1024 (1996) — the
  S-matrix recursion.
- R. C. Rumpf, "Improved formulation of scattering matrices for semi-analytical
  methods that is consistent with convention," *PIER B* **35**, 241 (2011) — the
  scattering-matrix conventions this implementation follows.

These are background for the *method*, not citations *of Ikarus*; cite the
software entry above for the implementation.
