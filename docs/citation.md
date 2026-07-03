# Citation

If Ikarus carried your results somewhere good, please cite it — it's how
open-source scientific software earns its keep.

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

Set `version` to the release you actually used —
`python -c "import ikarus; print(ikarus.__version__)"`.

!!! tip "Want a DOI?"
    For a permanently archived, versioned reference, connect the GitHub repo to
    [Zenodo](https://zenodo.org/) — each release then mints a DOI you can drop
    into the entry above.

## Background references { #background-references }

The method itself stands on classic shoulders. If you discuss RCWA in your
paper, these are the canonical citations:

- M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of
  planar-grating diffraction," *J. Opt. Soc. Am.* **71**, 811 (1981) — the
  original formulation.
- M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, "Formulation for
  stable and efficient implementation of the rigorous coupled-wave analysis of
  binary gratings," *J. Opt. Soc. Am. A* **12**, 1068 (1995).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic
  structures," *J. Opt. Soc. Am. A* **13**, 1870 (1996) — the factorization
  rules; the default normal-vector method applies Li's **inverse rule** along
  the local boundary normal.
- L. Li, "New formulation of the Fourier modal method for crossed surface-relief
  gratings," *J. Opt. Soc. Am. A* **14**, 2758 (1997) — the two-step factorization
  behind `factorization="li"`, to which the default reduces on axis-aligned
  geometry.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, "Normal vector method
  for convergence improvement using the RCWA for crossed gratings," *J. Opt.
  Soc. Am. A* **24**, 2880 (2007) — the normal-vector (Fast Fourier
  Factorization) method behind the default `factorization="auto"`.
- L. Li, "Formulation and comparison of two recursive matrix algorithms for
  modeling layered diffraction gratings," *J. Opt. Soc. Am. A* **13**, 1024
  (1996) — the S-matrix recursion.
- R. C. Rumpf, "Improved formulation of scattering matrices for
  semi-analytical methods that is consistent with convention," *PIER B*
  **35**, 241 (2011) — the scattering-matrix conventions Ikarus follows.

These cite the *method*; the software entry above cites the *implementation*.
