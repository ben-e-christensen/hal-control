"""Linear acoustic field of the array, with an image-source reflector.

Ported from refs/hal_lev_sim.py (piston sources, 2*J1(x)/x directivity,
Gor'kov potential) into the app frame: array at Y = 0 facing +Y, reflector
plate at Y = gap. The reflector is modelled by mirroring every emitter across
Y = gap (same phase, amplitude x reflection coefficient, facing -Y).

Phases fed in here should be what the FPGA actually emits: use
model_phases(codes) to get the quantized phases (code 32 = off -> no source).
Single wavelength everywhere: geometry.WAVELENGTH.
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import EMITTER_POS, K, WAVELENGTH
from .phases import OFF_CODE, codes_to_phases

F_HZ = 40e3
OMEGA = 2 * math.pi * F_HZ
C_AIR = WAVELENGTH * F_HZ          # 346 m/s, consistent with the phase solver
RHO_AIR = 1.2
PISTON_RADIUS = 4.5e-3             # m, effective radius of a 10 mm transducer
P0_PER_VOLT = 0.17                 # Pa*m per Vpp at 1 m (approximate)


# ---- directivity: 2*J1(x)/x, tabulated over sin(theta) in [0, 1] ---------------
def _j1_ratio(x):
    """2*J1(x)/x from its power series (x <= ~3.3 here; 20 terms is plenty)."""
    acc = np.zeros_like(x)
    x2 = x * x
    for m in reversed(range(20)):
        c = (-1) ** m / (math.factorial(m) * math.factorial(m + 1) * 2 ** (2 * m + 1))
        acc = acc * x2 + c
    return 2.0 * acc


_SIN_TABLE = np.linspace(0.0, 1.0, 1025)
_DIR_TABLE = _j1_ratio(K * PISTON_RADIUS * _SIN_TABLE)


def directivity(sin_theta):
    return np.interp(sin_theta, _SIN_TABLE, _DIR_TABLE)


# ---- model --------------------------------------------------------------------
@dataclass(frozen=True)
class FieldModel:
    gap: float                      # m, array face to reflector plate
    reflector: bool = True
    refl_coeff: float = 1.0
    vpp: float = 12.0               # drive voltage, only scales absolute values

    @property
    def p0(self):
        # square-wave drive: the 40 kHz fundamental is 4/pi of the square wave
        return P0_PER_VOLT * self.vpp * 4 / math.pi


@dataclass(frozen=True)
class Sources:
    pos: np.ndarray       # (M, 3)
    normal_y: np.ndarray  # (M,) +1 faces +Y, -1 faces -Y
    amp: np.ndarray       # (M,) complex: p0 * R * exp(i*phase)


def model_phases(codes):
    """FPGA codes -> the quantized phases actually emitted (NaN = off)."""
    return codes_to_phases(np.asarray(codes))


def make_sources(phases, model):
    """Real emitters (+ mirrored images if the reflector is on). NaN phases = off."""
    ph = np.asarray(phases, dtype=float)
    on = ~np.isnan(ph)
    pos = EMITTER_POS[on]
    amp = model.p0 * np.exp(1j * ph[on])
    normal = np.ones(len(pos))
    if model.reflector and model.refl_coeff != 0:
        img = pos.copy()
        img[:, 1] = 2 * model.gap - img[:, 1]
        pos = np.vstack([pos, img])
        amp = np.concatenate([amp, amp * model.refl_coeff])
        normal = np.concatenate([normal, -np.ones(len(img))])
    return Sources(pos, normal, amp)


_S2_TABLE_N = 2048
# directivity tabulated over sin^2(theta) so the kernel needs no extra sqrt
_DIR_S2 = directivity(np.sqrt(np.linspace(0.0, 1.0, _S2_TABLE_N + 1))).astype(np.float32)
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import os
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1)),
                                   thread_name_prefix="field")
    return _POOL


def _kernel(P, e, ny, ar, ai, want_grad):
    """float32 field kernel for one chunk of points. Returns p (n,), grad (n,3)|None."""
    dx = P[:, 0:1] - e[None, :, 0]
    dy = P[:, 1:2] - e[None, :, 1]
    dz = P[:, 2:3] - e[None, :, 2]
    r = dx * dx
    r += dz * dz
    r += dy * dy
    np.maximum(r, np.float32(1e-8), out=r)
    np.sqrt(r, out=r)
    inv = np.reciprocal(r)
    c = dy * inv                                   # cos(theta) w.r.t. +Y
    s2 = np.subtract(np.float32(1.0), c * c)
    idx = (s2 * _S2_TABLE_N).astype(np.int32)
    np.clip(idx, 0, _S2_TABLE_N, out=idx)
    w = _DIR_S2[idx]
    w *= inv
    w *= (c * ny[None, :]) > 0                     # nothing radiates backwards
    kr = r * np.float32(K)
    cs, sn = np.cos(kr), np.sin(kr)
    W, V = cs * w, sn * w
    p = (W @ ar - V @ ai) + 1j * (W @ ai + V @ ar)
    if not want_grad:
        return p, None
    # d/dx [w e^{ikr}] ~ w e^{ikr} (ik - 1/r) dx/r  (directivity derivative ignored)
    k = np.float32(K)
    Ar = (-(cs * inv) - k * sn) * w * inv
    Ai = (k * cs - sn * inv) * w * inv
    g = np.empty((len(P), 3), complex)
    for j, d in enumerate((dx, dy, dz)):
        Br, Bi = Ar * d, Ai * d
        g[:, j] = (Br @ ar - Bi @ ai) + 1j * (Br @ ai + Bi @ ar)
    return p, g


def pressure(sources, pts, want_grad=False, chunk_elems=100_000):
    """Complex pressure (Pa) at pts (N, 3); optionally its gradient (N, 3).

    float32 kernel (relative error ~1e-3, fine for display), chunked over
    threads. The gradient uses the monopole derivative (ik - 1/r) r_hat and
    ignores the directivity derivative, as in hal_lev_sim.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float32))
    n, m = len(pts), len(sources.pos)
    if m == 0:
        return np.zeros(n, complex), (np.zeros((n, 3), complex) if want_grad else None)
    e = sources.pos.astype(np.float32)
    ny = sources.normal_y.astype(np.float32)
    ar = sources.amp.real.astype(np.float32)
    ai = sources.amp.imag.astype(np.float32)
    step = max(1, chunk_elems // m // (3 if want_grad else 1))
    chunks = [pts[s:s + step] for s in range(0, n, step)]
    if len(chunks) == 1:
        res = [_kernel(chunks[0], e, ny, ar, ai, want_grad)]
    else:
        res = list(_pool().map(lambda P: _kernel(P, e, ny, ar, ai, want_grad), chunks))
    p = np.concatenate([r[0] for r in res])
    g = np.concatenate([r[1] for r in res]) if want_grad else None
    return p, g


def pressure_reference(sources, pts, want_grad=False):
    """Straightforward float64 version of pressure() (same model), for tests."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    e, ny, a = sources.pos, sources.normal_y, sources.amp
    dx = pts[:, 0:1] - e[None, :, 0]
    dy = pts[:, 1:2] - e[None, :, 1]
    dz = pts[:, 2:3] - e[None, :, 2]
    r = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-4)
    cos_t = ny[None, :] * dy / r
    D = directivity(np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))) * (cos_t > 0)
    term = a[None, :] * D / r * np.exp(1j * K * r)
    p = term.sum(axis=1)
    if not want_grad:
        return p, None
    common = term * (1j * K - 1.0 / r) / r
    g = np.stack([(common * d).sum(axis=1) for d in (dx, dy, dz)], axis=1)
    return p, g


# ---- Gor'kov potential ------------------------------------------------------------
@dataclass(frozen=True)
class Bead:
    diameter: float = 2.0e-3        # m
    rho: float = 29.0               # kg/m^3 (EPS)
    c: float = 900.0                # m/s

    @property
    def volume(self):
        return 4 / 3 * math.pi * (self.diameter / 2) ** 3

    @property
    def k1(self):
        f1 = 1 - (RHO_AIR * C_AIR ** 2) / (self.rho * self.c ** 2)
        return self.volume * f1 / (4 * RHO_AIR * C_AIR ** 2)

    @property
    def k2(self):
        f2 = 2 * (self.rho - RHO_AIR) / (2 * self.rho + RHO_AIR)
        return self.volume * f2 * 3 / (8 * OMEGA ** 2 * RHO_AIR)


def gorkov(sources, pts, bead=Bead()):
    """Gor'kov potential U (J) at pts, no gravity."""
    p, g = pressure(sources, pts, want_grad=True)
    return bead.k1 * np.abs(p) ** 2 - bead.k2 * (np.abs(g) ** 2).sum(axis=1)


# ---- slices -------------------------------------------------------------------------
@dataclass(frozen=True)
class Slice:
    h: np.ndarray        # horizontal axis coordinates (m)
    v: np.ndarray        # vertical axis coordinates (m)
    values: np.ndarray   # (len(v), len(h)) real: |p| in Pa, or U in J


def side_slice(sources, gap, z, nx, ny, half_width=0.095, what="amplitude", bead=Bead()):
    """XY plane at fixed Z. h = X, v = Y (array face to plate)."""
    x = np.linspace(-half_width, half_width, nx)
    y = np.linspace(min(1e-3, gap / 4), gap - min(0.5e-3, gap / 8), ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return Slice(x, y, _eval(sources, pts, what, bead).reshape(X.shape))


def top_slice(sources, y, nx, nz, half_width=0.095, what="amplitude", bead=Bead()):
    """XZ plane at fixed Y. h = X, v = Z."""
    x = np.linspace(-half_width, half_width, nx)
    z = np.linspace(-half_width, half_width, nz)
    X, Z = np.meshgrid(x, z)
    pts = np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()])
    return Slice(x, z, _eval(sources, pts, what, bead).reshape(X.shape))


def _eval(sources, pts, what, bead):
    if what == "gorkov":
        return gorkov(sources, pts, bead)
    p, _ = pressure(sources, pts)
    return np.abs(p)


def pressure_nodes(sources, gap, x, z, n=500, ring_radius=2.5e-3, min_dip=0.15):
    """Y positions of standing-wave pressure nodes near the line (x, *, z).

    |p|^2 is averaged over the line plus 8 points on a ring of ring_radius
    around it, so a vortex's silent core doesn't produce spurious nodes. A
    minimum counts only if it dips at least min_dip (fraction) below the
    lower of the maxima within lambda/2 on either side. With the reflector on
    these sit roughly lambda/2 apart; small light beads usually trap there.
    """
    y = np.linspace(0.5e-3, gap - 0.2e-3, n)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    offs = [(0.0, 0.0)] + [(ring_radius * np.cos(t), ring_radius * np.sin(t)) for t in ang]
    pts = np.concatenate([np.column_stack([np.full(n, x + ox), y, np.full(n, z + oz)])
                          for ox, oz in offs])
    p2 = np.abs(pressure(sources, pts)[0]).reshape(len(offs), n) ** 2
    a = np.sqrt(p2.mean(axis=0))
    dy = y[1] - y[0]
    w = max(2, int(round(WAVELENGTH / 2 / dy)))
    out = []
    for k in range(1, n - 1):
        if not (a[k] < a[k - 1] and a[k] <= a[k + 1]):
            continue
        left, right = a[max(0, k - w):k].max(), a[k + 1:k + 1 + w].max()
        ref = min(left, right)
        if ref <= 0 or (ref - a[k]) < min_dip * ref:
            continue
        den = a[k - 1] - 2 * a[k] + a[k + 1]
        off = 0.5 * (a[k - 1] - a[k + 1]) / den if den > 0 else 0.0
        out.append(y[k] + off * dy)
    return np.array(out)


def is_off(codes):
    return np.asarray(codes) == OFF_CODE
