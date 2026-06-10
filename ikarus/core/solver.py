r"""Core RCWA linear algebra: eigenmodes, scattering matrices, cascade.

This module contains the numerically heavy, *stateless* heart of the method.  It
implements the scattering-matrix formulation of rigorous coupled-wave analysis
for 2-D periodic (crossed-grating) structures, following the convolution-matrix
eigenmode approach (Moharam/Gaylord, generalized by Li; presentation after
Rumpf).  The scattering-matrix cascade (as opposed to a transfer-matrix product)
is unconditionally stable for thick and evanescent layers.

All wavevectors are normalized by the vacuum wavenumber ``k0 = 2*pi/lambda``.
Within a layer the tangential field amplitudes obey

    d^2/dz'^2 [Sx; Sy] = Omega^2 [Sx; Sy],   Omega^2 = P Q,   z' = k0 z,

with the block matrices ``P`` and ``Q`` built from the permittivity convolution
matrix and the diagonal wavevector matrices ``Kx, Ky``.  The eigenpairs of
``Omega^2`` give the layer modes; each layer's scattering matrix is referenced to
a common free-space *gap medium* and the layers + two semi-infinite regions are
combined with the Redheffer star product.

Time convention: ``exp(-i omega t)`` (absorbing media have ``Im(eps) > 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import scipy.linalg as _sla

from .fourier import HarmonicGrid, convolution_matrix


def _inv(matrix: np.ndarray) -> np.ndarray:
    return _sla.inv(matrix, check_finite=False)


def _rdiv(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Right division ``M @ inv(K)`` via a solve (never forms ``inv(K)``)."""
    return _sla.solve(K.T, M.T, check_finite=False).T


def _block(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Assemble a 2x2 block matrix ``[[a, b], [c, d]]``."""
    return np.block([[a, b], [c, d]])


@dataclass
class SMatrix:
    """A scattering matrix with four ``2P x 2P`` blocks."""

    S11: np.ndarray
    S12: np.ndarray
    S21: np.ndarray
    S22: np.ndarray

    @classmethod
    def identity(cls, dim: int) -> "SMatrix":
        """Star-product identity element (a zero-thickness gap)."""
        z = np.zeros((dim, dim), dtype=complex)
        i = np.eye(dim, dtype=complex)
        return cls(S11=z.copy(), S12=i.copy(), S21=i.copy(), S22=z.copy())


def redheffer_star(a: SMatrix, b: SMatrix) -> SMatrix:
    """Redheffer star product ``a ⋆ b`` (``a`` on top/left, ``b`` below/right)."""
    dim = a.S11.shape[0]
    I = np.eye(dim, dtype=complex)
    D = _rdiv(a.S12, I - b.S11 @ a.S22)
    F = _rdiv(b.S21, I - a.S22 @ b.S11)
    S11 = a.S11 + D @ b.S11 @ a.S21
    S12 = D @ b.S12
    S21 = F @ a.S21
    S22 = b.S22 + F @ a.S22 @ b.S12
    return SMatrix(S11, S12, S21, S22)


# ---------------------------------------------------------------------------
# Wavevector matrices and reference (gap) medium
# ---------------------------------------------------------------------------
def wavevector_matrices(
    grid: HarmonicGrid,
    kx0: complex,
    ky0: complex,
    period_x: float,
    period_y: float,
    wavelength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal normalized wavevector matrices ``Kx, Ky`` over all harmonics."""
    p, q = grid.index_arrays()
    # Normalized reciprocal-lattice shifts: (lambda / period) * order.
    kx = kx0 - p * (wavelength / period_x)
    ky = ky0 - q * (wavelength / period_y)
    return np.diag(kx.astype(complex)), np.diag(ky.astype(complex))


def uniform_modes(eps: complex, Kx: np.ndarray, Ky: np.ndarray):
    """Analytic eigenmodes ``(W, V, Kz)`` of a homogeneous medium.

    Used for the reference gap medium and for the two semi-infinite regions.
    Crucially, the modal eigenvalues ``lam`` are obtained with the *same*
    :func:`_forward_branch` selection used for patterned layers, so the magnetic
    eigenmodes ``V`` are sign-consistent across gap, regions and layers.

    Building ``lam`` directly from a per-order ``conj(sqrt(conj(.)))`` (the old
    approach) is subtly wrong: for a lossless medium an evanescent order's
    argument is *real and negative*, and the signed-zero produced by the double
    conjugation lands ``sqrt`` on the opposite branch -- flipping the sign of the
    evanescent ``V`` columns.  That error is invisible whenever only the zero
    order propagates (e.g. Fresnel slabs) but corrupts every diffraction grating.
    """
    p = Kx.shape[0]
    kx, ky = np.diag(Kx), np.diag(Ky)
    # Modal eigenvalue per order: lam^2 = -(eps - kx^2 - ky^2).  Forward branch.
    lam_order = _forward_branch(kx ** 2 + ky ** 2 - eps)
    lam = np.concatenate([lam_order, lam_order])
    W = np.eye(2 * p, dtype=complex)
    # Every Q block is diagonal for a homogeneous medium -- build from vectors.
    Q = _block(np.diag(kx * ky), np.diag(eps - kx * kx),
               np.diag(ky * ky - eps), np.diag(-ky * kx))
    V = Q / lam[np.newaxis, :]   # Q @ diag(1/lam)
    # Physical longitudinal wavevector: lam = i*kz for propagating, so kz = -i*lam
    # (Re(kz) > 0 for propagating orders, 0 for evanescent -- used in the energy
    # balance weighting and field reconstruction).
    Kz = np.diag(-1j * lam_order)
    return W, V, Kz


# Negligible loss added to every medium so that no diffraction order can land
# *exactly* on a light line (Rayleigh/Wood anomaly), which would make the modal
# Q matrix and eigenvalues singular.  The induced energy defect is ~1e-9.
_ANOMALY_LOSS = 1e-9j


def free_space_modes(Kx: np.ndarray, Ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenmodes ``(W0, V0)`` of the vacuum gap medium used as reference."""
    W0, V0, _ = uniform_modes(1.0 - _ANOMALY_LOSS, Kx, Ky)
    return W0, V0


# ---------------------------------------------------------------------------
# Layer modes and scattering matrix
# ---------------------------------------------------------------------------
@dataclass
class LayerModes:
    """Eigenmode data for one layer (reused for field reconstruction)."""

    W: np.ndarray
    V: np.ndarray
    lam: np.ndarray  # 1-D array of 2P eigenvalues (the modal kz', complex)
    k0L: float  # normalized thickness k0 * height
    ERC: np.ndarray = None  # permittivity convolution matrix (engineering conv.)


def _forward_branch(eigvals: np.ndarray) -> np.ndarray:
    """Select the forward / outgoing modal eigenvalues ``lam = sqrt(eigvals)``.

    The second-order modal problem has eigenvalues in ``+/-`` pairs; we must keep
    the *forward* one consistently.  A forward mode either decays in ``+z``
    (``Re(lam) > 0``) or, when it is (near) lossless and propagating
    (``Re(lam) ~ 0``), advances with ``Im(lam) > 0``.

    The naive ``Re(lam) >= 0`` rule alone is *not enough*: ``np.linalg.eig`` of the
    non-symmetric ``P Q`` returns a tiny spurious imaginary part on the squared
    eigenvalue of a propagating mode, which sends ``sqrt`` onto the ``-i`` branch
    (``Re`` stays ~0, so the real-part test never fires).  That single
    wrong-branch mode silently corrupts the scattering matrix of any patterned
    layer (uniform layers have exactly-real squared eigenvalues and are immune).
    """
    lam = np.sqrt(np.asarray(eigvals, dtype=complex))
    marginal = np.abs(lam.real) <= 1e-8 * np.abs(lam)
    flip = (lam.real < 0) & ~marginal
    flip |= marginal & (lam.imag < 0)
    lam = np.where(flip, -lam, lam)
    return _regularize_grazing(lam)


def _regularize_grazing(lam: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Nudge eigenvalues away from exactly zero (Rayleigh/Wood anomalies).

    A diffraction order grazing the light line gives ``lam = 0``, which makes the
    magnetic eigenmodes ``V = Q W / LAM`` singular.  Such orders carry vanishing
    longitudinal power, so a tiny forward-decaying nudge leaves the physics
    essentially unchanged while keeping the linear algebra well posed.
    """
    tiny = np.abs(lam) < eps
    if np.any(tiny):
        lam = lam.copy()
        lam[tiny] = eps * (1.0 + 1j)
    return lam


def layer_modes(ERC: np.ndarray, Kx: np.ndarray, Ky: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenmodes ``(W, V, lam)`` of a layer with permittivity conv. matrix ERC.

    ``ERC`` may be a full (patterned/anisotropic) matrix or ``eps * I`` for a
    uniform layer.  Returns the electric eigenvectors ``W``, magnetic
    eigenvectors ``V`` and the 1-D array of modal eigenvalues ``lam`` (with
    ``Re(lam) >= 0`` so ``exp(-lam k0 L)`` is bounded for forward modes).
    """
    p = Kx.shape[0]
    I = np.eye(p, dtype=complex)
    Einv = _inv(ERC)
    kx, ky = np.diag(Kx)[:, None], np.diag(Ky)[None, :]
    kxc, kyr = np.diag(Kx), np.diag(Ky)  # 1-D, for the diagonal Q blocks

    # Kx/Ky are diagonal: their products with Einv are row/column scalings, and
    # the Q blocks are diagonal -- avoid the dense O(N^3) matmuls.
    P = _block(kx * Einv * ky, I - kx * Einv * kx.T,
               (ky.T * Einv * ky) - I, -ky.T * Einv * kx.T)
    Q = _block(np.diag(kxc * kyr), ERC - np.diag(kxc * kxc),
               np.diag(kyr * kyr) - ERC, -np.diag(kyr * kxc))

    omega2 = P @ Q
    eigvals, W = _sla.eig(omega2, overwrite_a=True, check_finite=False)
    lam = _forward_branch(eigvals)
    V = (Q @ W) / lam[np.newaxis, :]   # Q @ W @ diag(1/lam)
    return W, V, lam


def layer_smatrix(
    ERC: np.ndarray,
    Kx: np.ndarray,
    Ky: np.ndarray,
    k0L: float,
    V0: np.ndarray,
) -> tuple[SMatrix, LayerModes]:
    """Scattering matrix of a single finite layer, referenced to the gap medium.

    The gap electric modes ``W0`` are the identity, so ``W^{-1} W0 = W^{-1}``.
    The phase matrix ``X = exp(-lam k0 L)`` is diagonal and applied as a
    row-broadcast rather than a dense matmul.
    """
    W, V, lam = layer_modes(ERC, Kx, Ky)

    Winv = _inv(W)
    VinvV0 = _inv(V) @ V0
    A = Winv + VinvV0
    B = Winv - VinvV0
    x = np.exp(-lam * k0L)[:, np.newaxis]   # X @ M == x * M
    Ainv = _inv(A)

    XB = x * B
    fac = _inv(A - XB @ Ainv @ XB)
    S11 = fac @ (XB @ Ainv @ (x * A) - B)
    S12 = fac @ (x * (A - B @ Ainv @ B))
    smat = SMatrix(S11=S11, S12=S12, S21=S12.copy(), S22=S11.copy())
    return smat, LayerModes(W=W, V=V, lam=lam, k0L=k0L, ERC=ERC)


# ---------------------------------------------------------------------------
# Semi-infinite region scattering matrices
# ---------------------------------------------------------------------------
def reflection_smatrix(eps_ref, Kx, Ky, V0inv):
    """Scattering matrix coupling the gap medium to the cover (reflection) region.

    The region and gap electric modes are both the identity, so
    ``A = I + V0^{-1} V`` and ``B = I - V0^{-1} V`` (``V0inv`` is precomputed once).
    """
    _, V, Kz = uniform_modes(eps_ref, Kx, Ky)
    M = V0inv @ V
    I = np.eye(M.shape[0], dtype=complex)
    A, B = I + M, I - M
    Ainv = _inv(A)
    S = SMatrix(
        S11=-Ainv @ B,
        S12=2 * Ainv,
        S21=0.5 * (A - B @ Ainv @ B),
        S22=B @ Ainv,
    )
    return S, Kz


def transmission_smatrix(eps_trn, Kx, Ky, V0inv):
    """Scattering matrix coupling the gap medium to the substrate (transmission)."""
    _, V, Kz = uniform_modes(eps_trn, Kx, Ky)
    M = V0inv @ V
    I = np.eye(M.shape[0], dtype=complex)
    A, B = I + M, I - M
    Ainv = _inv(A)
    S = SMatrix(
        S11=B @ Ainv,
        S12=0.5 * (A - B @ Ainv @ B),
        S21=2 * Ainv,
        S22=-Ainv @ B,
    )
    return S, Kz


# ---------------------------------------------------------------------------
# Top-level field solve
# ---------------------------------------------------------------------------
@dataclass
class FieldSolution:
    """Outgoing fields and diffraction efficiencies of a solved stack."""

    grid: HarmonicGrid
    # Complex tangential + longitudinal field amplitudes per harmonic.
    rx: np.ndarray
    ry: np.ndarray
    rz: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    tz: np.ndarray
    # Per-order diffraction efficiencies (real, sum to <= 1).
    R_orders: np.ndarray
    T_orders: np.ndarray
    Kz_ref: np.ndarray
    Kz_trn: np.ndarray
    kz_inc: complex
    # Bookkeeping reused downstream (fields, angles).
    Kx: np.ndarray
    Ky: np.ndarray
    eps_ref: complex
    eps_trn: complex
    global_smatrix: SMatrix
    layer_modes: list
    # Data required to reconstruct internal fields (set by solve_stack).
    s_ref: SMatrix = None
    s_trn: SMatrix = None
    layer_smatrices: list = None
    W0: np.ndarray = None
    V0: np.ndarray = None
    c_src: np.ndarray = None
    heights: tuple = ()
    k0: float = 0.0
    period_x: float = 0.0
    period_y: float = 0.0

    @property
    def R_total(self) -> float:
        return float(np.sum(self.R_orders))

    @property
    def T_total(self) -> float:
        return float(np.sum(self.T_orders))


def solve_stack(
    eps_grids: list[np.ndarray],
    heights: list[float],
    eps_ref: complex,
    eps_trn: complex,
    grid: HarmonicGrid,
    kx0: complex,
    ky0: complex,
    period_x: float,
    period_y: float,
    wavelength: float,
    polarization_xy: tuple[complex, complex],
) -> FieldSolution:
    """Solve a layered stack and return outgoing fields + efficiencies.

    Parameters
    ----------
    eps_grids, heights:
        Real-space permittivity samples (``(Nx, Ny)`` arrays) and thicknesses of
        the *interior* (finite) layers, ordered cover-side to substrate-side.
        Permittivities follow the physics ``exp(-i w t)`` convention used by the
        rest of Ikarus: absorbing media have ``Im(eps) > 0``.
    eps_ref, eps_trn:
        Scalar permittivities of the semi-infinite cover and substrate.
    grid:
        Harmonic truncation.
    kx0, ky0:
        Normalized in-plane wavevector of the incident wave.
    polarization_xy:
        Tangential ``(p_x, p_y)`` components of the incident polarization vector.

    Notes
    -----
    The linear-algebra core is written in the engineering ``exp(+j w t)``
    convention (loss = ``Im(eps) < 0``).  This routine is the single bridge to
    the package's physics convention: it conjugates the permittivities on the way
    in and conjugates the complex field amplitudes on the way out, exploiting the
    identity ``Sol_phys(eps) = conj( Sol_eng(conj eps) )``.  Energy efficiencies
    are conjugation-invariant.
    """
    Kx, Ky = wavevector_matrices(grid, kx0, ky0, period_x, period_y, wavelength)
    W0, V0 = free_space_modes(Kx, Ky)
    V0inv = _inv(V0)   # gap admittance inverse, reused by both regions
    k0 = 2.0 * np.pi / wavelength

    # --- physics -> engineering convention bridge (conjugate permittivities).
    # A negligible loss (_ANOMALY_LOSS) regularizes exact Rayleigh anomalies.
    eps_ref_e = np.conj(eps_ref) - _ANOMALY_LOSS
    eps_trn_e = np.conj(eps_trn) - _ANOMALY_LOSS
    erc_list = [convolution_matrix(np.conj(g) - _ANOMALY_LOSS, grid) for g in eps_grids]

    # Assemble the global scattering matrix: cover ⋆ layers ⋆ substrate.
    S_ref, Kz_ref = reflection_smatrix(eps_ref_e, Kx, Ky, V0inv)
    S = S_ref
    modes: list[LayerModes] = []
    layer_smats: list[SMatrix] = []
    for ERC, h in zip(erc_list, heights):
        S_layer, lm = layer_smatrix(ERC, Kx, Ky, k0 * h, V0)
        modes.append(lm)
        layer_smats.append(S_layer)
        S = redheffer_star(S, S_layer)
    S_trn, Kz_trn = transmission_smatrix(eps_trn_e, Kx, Ky, V0inv)
    S = redheffer_star(S, S_trn)

    # Incident-mode coefficients: unit amplitude in the (0,0) order.  The
    # polarization is part of the physics->engineering bridge and must be
    # conjugated like the permittivities (matters only for complex/circular
    # polarization; real linear vectors are unchanged).
    P = grid.size
    delta = np.zeros(P, dtype=complex)
    delta[grid.zero_order_index()] = 1.0
    px, py = np.conj(polarization_xy[0]), np.conj(polarization_xy[1])
    c_src = np.concatenate([px * delta, py * delta])  # W_ref = I

    c_ref = S.S11 @ c_src
    c_trn = S.S21 @ c_src
    rx, ry = c_ref[:P], c_ref[P:]
    tx, ty = c_trn[:P], c_trn[P:]
    # Kx, Ky, Kz are all diagonal -> longitudinal components are pure elementwise.
    kx, ky = np.diag(Kx), np.diag(Ky)
    kzr, kzt = np.diag(Kz_ref), np.diag(Kz_trn)
    rz = -(kx * rx + ky * ry) / kzr
    tz = -(kx * tx + ky * ty) / kzt

    # Diffraction efficiencies (mu = 1): |field|^2 weighted by Re(kz)/Re(kz_inc).
    kz_inc = np.sqrt(eps_ref_e - kx0**2 - ky0**2 + 0j)
    if kz_inc.real < 0:
        kz_inc = -kz_inc
    R2 = np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2
    T2 = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2
    R_orders = R2 * np.real(kzr) / np.real(kz_inc)
    T_orders = T2 * np.real(kzt) / np.real(kz_inc)

    # Conjugate complex amplitudes back to the physics convention.
    rx, ry, rz = np.conj(rx), np.conj(ry), np.conj(rz)
    tx, ty, tz = np.conj(tx), np.conj(ty), np.conj(tz)

    return FieldSolution(
        grid=grid, rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz,
        R_orders=R_orders, T_orders=T_orders,
        Kz_ref=Kz_ref, Kz_trn=Kz_trn, kz_inc=kz_inc,
        Kx=Kx, Ky=Ky, eps_ref=eps_ref, eps_trn=eps_trn,
        global_smatrix=S, layer_modes=modes,
        s_ref=S_ref, s_trn=S_trn, layer_smatrices=layer_smats,
        W0=W0, V0=V0, c_src=c_src, heights=tuple(heights), k0=k0,
        period_x=period_x, period_y=period_y,
    )
