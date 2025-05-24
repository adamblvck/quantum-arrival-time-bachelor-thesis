from __future__ import annotations
import numpy as np
from typing import Literal, Optional

# --------------------------------------------------------------------
# Physical constants (define your own unit system as needed)
# --------------------------------------------------------------------
hbar: float = 1.0  # Planck constant / (2π)
m: float = 1.0     # mass

###############################################################################
#  Absorber infrastructure
###############################################################################
class Absorber:
    """Hold CAP/mask data and track cumulative absorption."""

    def __init__(self, *, cap: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None) -> None:
        self.cap = cap          # −i W(x) term added to V once
        self.mask = mask        # multiplicative mask applied each full step
        self.initial_norm: Optional[float] = None
        self.absorbed_cum: float = 0.0  # probability lost so far

    # ------------------------------------------------------------------
    def begin(self, psi: np.ndarray, dz: float) -> None:
        """Call once *before* time propagation begins."""
        self.initial_norm = float(np.sum(np.abs(psi) ** 2) * dz)
        # ensure numerical noise never yields div/0
        if self.initial_norm == 0:
            raise ValueError("Initial norm is zero; check initial wave‑packet.")

    # ------------------------------------------------------------------
    def update(self, psi: np.ndarray, dz: float) -> float:
        """Update cumulative loss; return absorbed fraction so far."""
        if self.initial_norm is None:
            raise RuntimeError("Absorber.begin() must be called first")
        current_norm = float(np.sum(np.abs(psi) ** 2) * dz)
        self.absorbed_cum = self.initial_norm - current_norm
        return self.absorbed_cum / self.initial_norm

###############################################################################
#  Absorber factory
###############################################################################

def _xi_array(z: np.ndarray, x_min: float, x_max: float, width: float) -> np.ndarray:
    """Distance ξ(x) into the absorbing layer, 0 in the interior."""
    xi = np.zeros_like(z)
    left_region = z < (x_min + width)
    xi[left_region] = (x_min + width) - z[left_region]
    right_region = z > (x_max - width)
    # xi[right_region] = z[right_region] - (x_max - width)
    return xi


def build_absorber(
    z: np.ndarray,
    *,
    absorber_type: Literal[
        "none",
        "poly_cap",
        "gauss_cap",
        "cosine_mask",
        "tfap_cap",
        "rf_cap",
        "ses_cap",
    ] = "none",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    width: float = 6.0,
    # --- polynomial CAP
    n: int = 4,
    eta: float = 1.0,
    # --- Gaussian
    alpha: float = 2.5,
    # --- cosine mask
    p: int = 4,
    # --- TFAP
    emin: float = 0.02,
    # --- RF‑CAP
    eta_rf: float = 1.0,
    # --- SES
    theta: float = 0.35,
) -> Absorber:
    """Construct an :class:`Absorber` for the selected *absorber_type*."""
    if absorber_type == "none":
        return Absorber()

    if x_min is None:
        x_min = float(z[0])
    if x_max is None:
        x_max = float(z[-1])

    xi = _xi_array(z, x_min, x_max, width)

    # ===== polynomial CAP =================================================
    if absorber_type == "poly_cap":
        print(f"eta: {eta}, n: {n}")
        W = eta * (xi / width) ** n
        return Absorber(cap=W.astype(float))

    # ===== Gaussian CAP / mask ============================================
    if absorber_type == "gauss_cap":
        W = eta * np.exp(-alpha * (xi / width) ** 2)
        return Absorber(cap=W.astype(float))

    # ===== cosine‑window mask =============================================
    if absorber_type == "cosine_mask":
        mask = np.ones_like(z)
        idx = xi > 0
        mask[idx] = np.sin(0.5 * np.pi * xi[idx] / width) ** p
        return Absorber(mask=mask.astype(float))

    # ===== Transmission‑Free (TFAP) CAP ===================================
    if absorber_type == "tfap_cap":
        beta = np.sqrt(2.0 * m * emin) / hbar
        W = (hbar ** 2) * beta ** 2 / (4.0 * m) / np.cosh(beta * xi) ** 2
        return Absorber(cap=W.astype(float))

    # ===== Reflection‑Free (RF‑CAP) =======================================
    if absorber_type == "rf_cap":
        s = xi / width
        W = eta_rf * s ** 2 * (1.0 - 2.0 / 3.0 * s)
        return Absorber(cap=W.astype(float))

    # ===== Smooth Exterior‑Scaling (SES) CAP ==============================
    if absorber_type == "ses_cap":
        s = xi / width
        g = np.sin(0.5 * np.pi * np.clip(s, 0, 1)) ** 2
        W = 0.5 * theta ** 2 * g ** 2 / (width ** 2 / 4)
        return Absorber(cap=W.astype(float))

    raise ValueError(f"Unknown absorber_type '{absorber_type}'.")