"""Phase recipes (radians, FPGA channel order) and discretization to FPGA codes.

All recipes take a point in the app frame (metres) and return a float array
of shape (256,) in [0, 2*pi). NaN means "transducer off".

Sign conventions (locked by tests/test_phases.py):
  focus  : phi_i = -k * |e_i - p|  (mod 2*pi), same as SonicSurface.phasesForFocus
  twin   : focus + pi on emitters where cos(a)*(X - Xp) + sin(a)*(Z - Zp) > 0,
           i.e. the split line passes through the trap's lateral position and
           angle a = 0 puts the pi on the +X side.
  vortex : focus + charge * atan2(X - Xp, Z - Zp) + rotation, the same
           handedness as SonicSurface.vortexAt but centred on the trap.
           (hal_lev_sim uses atan2(dZ, dX): its charge +1 is our charge -1.)
"""

import numpy as np

from .geometry import EMITTER_POS, K, N_EMITTERS, PHASE_DIVS

TWO_PI = 2 * np.pi
OFF_CODE = PHASE_DIVS  # 32


def _point(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {p.shape}")
    return p


def focus(p):
    p = _point(p)
    d = np.linalg.norm(EMITTER_POS - p, axis=1)
    return np.mod(-K * d, TWO_PI)


def twin(p, angle=0.0):
    """Twin trap. angle (radians) rotates the signature split about the Y axis."""
    p = _point(p)
    dx = EMITTER_POS[:, 0] - p[0]
    dz = EMITTER_POS[:, 2] - p[2]
    side = (np.cos(angle) * dx + np.sin(angle) * dz) > 0
    return np.mod(focus(p) + np.pi * side, TWO_PI)


def vortex(p, charge=1, rotation=0.0):
    """Vortex trap with topological charge `charge`, centred on the trap's X/Z."""
    p = _point(p)
    dx = EMITTER_POS[:, 0] - p[0]
    dz = EMITTER_POS[:, 2] - p[2]
    return np.mod(focus(p) + charge * np.arctan2(dx, dz) + rotation, TWO_PI)


def all_on():
    """Every transducer on at phase 0 (plain standing wave against the plate)."""
    return np.zeros(N_EMITTERS)


def all_off():
    return np.full(N_EMITTERS, np.nan)


def discretize(phases):
    """Radians -> uint8 FPGA codes 0..31 (NaN -> 32 = off).

    Rounds to the nearest step with halves rounding up, like Ultraino's
    Java Math.round(phase / 2pi * 32) mod 32. (np.round would round
    halves to even, which differs on exact ties.)
    """
    ph = np.asarray(phases, dtype=float)
    if ph.shape != (N_EMITTERS,):
        raise ValueError(f"phases must have shape ({N_EMITTERS},), got {ph.shape}")
    off = np.isnan(ph)
    steps = np.mod(np.where(off, 0.0, ph), TWO_PI) * (PHASE_DIVS / TWO_PI)
    codes = np.mod(np.floor(steps + 0.5), PHASE_DIVS).astype(np.uint8)
    codes[off] = OFF_CODE
    return codes


def codes_to_phases(codes):
    """FPGA codes -> radians (32 -> NaN). Inverse of discretize up to quantization."""
    c = np.asarray(codes)
    ph = c.astype(float) * (TWO_PI / PHASE_DIVS)
    ph[c == OFF_CODE] = np.nan
    return ph
