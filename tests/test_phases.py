import numpy as np
import pytest

from hal_control import geometry as g
from hal_control import phases as ph
from helpers import wrap_diff

STEP = 2 * np.pi / 32


def codes_of_steps(steps):
    """discretize() of a 256-vector whose first len(steps) entries are given in steps."""
    v = np.zeros(256)
    v[: len(steps)] = np.asarray(steps, dtype=float) * STEP
    return ph.discretize(v)[: len(steps)].tolist()


# ---- discretize ---------------------------------------------------------------
def test_discretize_exact_steps():
    assert codes_of_steps(range(32)) == list(range(32))


def test_discretize_rounds_halves_up_like_java_math_round():
    # np.round would give 0, 2, 2 for 0.5, 1.5, 2.5 (half to even)
    assert codes_of_steps([0.5, 1.5, 2.5, 0.4999, 0.5001]) == [1, 2, 3, 0, 1]


def test_discretize_wraps():
    assert codes_of_steps([31.4999, 31.5, 32.0, 33.2, -0.4, -0.6, -32.0, 64.4]) == [
        31, 0, 0, 1, 0, 31, 0, 0]


def test_discretize_nan_is_off_and_dtype():
    v = np.zeros(256)
    v[[3, 200]] = np.nan
    c = ph.discretize(v)
    assert c.dtype == np.uint8
    assert c[3] == 32 and c[200] == 32
    assert np.all(np.delete(c, [3, 200]) == 0)
    assert c.max() <= 32


def test_discretize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ph.discretize(np.zeros(255))


def test_codes_to_phases_round_trip():
    c = np.arange(256) % 33
    c = c.astype(np.uint8)
    np.testing.assert_array_equal(ph.discretize(ph.codes_to_phases(c)), c)


def test_all_on_all_off():
    assert np.all(ph.discretize(ph.all_on()) == 0)
    assert np.all(ph.discretize(ph.all_off()) == 32)


# ---- focus ----------------------------------------------------------------------
POINTS = [np.array(p) / 1000 for p in [(0, 80, 0), (12.5, 40, -30), (-50, 120, 20), (3, 15, 7)]]


@pytest.mark.parametrize("p", POINTS)
def test_focus_arrives_in_phase(p):
    d = np.linalg.norm(g.EMITTER_POS - p, axis=1)
    arrival = ph.focus(p) + g.K * d
    np.testing.assert_allclose(wrap_diff(arrival, 0.0), 0.0, atol=1e-9)
    assert np.all((ph.focus(p) >= 0) & (ph.focus(p) < 2 * np.pi))


@pytest.mark.parametrize("p", POINTS)
def test_focus_matches_sonicsurface_continuous(ss, p):
    ref = ss.phasesForFocus(g.app_to_ss(p))
    np.testing.assert_allclose(wrap_diff(ph.focus(p), ref), 0.0, atol=1e-9)


# ---- twin -----------------------------------------------------------------------
def twin_mask(p, angle):
    d = wrap_diff(ph.twin(p, angle), ph.focus(p))
    on_pi = np.isclose(np.abs(d), np.pi, atol=1e-9)
    assert np.all(on_pi | np.isclose(d, 0.0, atol=1e-9))
    return on_pi


def test_twin_centre_splits_array_at_x_zero():
    p = np.array([0, 0.08, 0])
    X, Z = g.EMITTER_POS[:, 0], g.EMITTER_POS[:, 2]
    np.testing.assert_array_equal(twin_mask(p, 0.0), X > 0)
    np.testing.assert_array_equal(twin_mask(p, np.pi / 2), Z > 0)
    np.testing.assert_array_equal(twin_mask(p, np.pi), X < 0)
    assert twin_mask(p, 0.0).sum() == 128


def test_twin_split_follows_trap():
    p = np.array([0.02, 0.08, -0.01])
    np.testing.assert_array_equal(twin_mask(p, 0.0), g.EMITTER_POS[:, 0] > 0.02)
    np.testing.assert_array_equal(twin_mask(p, np.pi / 2), g.EMITTER_POS[:, 2] > -0.01)


def test_twin_matches_hal_lev_sim(sim_module):
    gap = 0.12
    p_sim = np.array([0.0205, -0.013, 0.05])      # x_s, y_s, z_s
    p = g.sim_to_app(p_sim, gap)
    sim_mask = np.isclose(np.abs(wrap_diff(
        sim_module.focus_phases(p_sim, gap, "twin trap"),
        sim_module.focus_phases(p_sim, gap, "focus"))), np.pi)
    np.testing.assert_array_equal(twin_mask(p, 0.0), g.sim_raster_to_fpga(sim_mask))


# ---- vortex ---------------------------------------------------------------------
@pytest.mark.parametrize("charge,rotation", [(1, 0.0), (2, 0.0), (-1, 0.0), (1, 0.7)])
def test_vortex_matches_sonicsurface_at_array_centre(ss, charge, rotation):
    captured = []
    ss.sendPhases = captured.append
    p = np.array([0.0, 0.07, 0.0])
    s = g.app_to_ss(p)
    ss.vortexAt(s[0], s[1], s[2], topoCharge=charge, rotation=rotation)
    np.testing.assert_allclose(
        wrap_diff(ph.vortex(p, charge, rotation), captured[0]), 0.0, atol=1e-4)


def test_vortex_is_centred_on_trap():
    p = np.array([0.03, 0.07, -0.02])
    off = wrap_diff(ph.vortex(p, 1), ph.focus(p))
    dx = g.EMITTER_POS[:, 0] - p[0]
    dz = g.EMITTER_POS[:, 2] - p[2]
    np.testing.assert_allclose(wrap_diff(off, np.arctan2(dx, dz)), 0.0, atol=1e-9)


def test_vortex_handedness_is_opposite_to_hal_lev_sim(sim_module):
    """hal_lev_sim adds atan2(dy_s, dx_s) = pi/2 - atan2(dX, dZ): its +1 is our -1."""
    gap = 0.12
    p_sim = np.array([0.011, 0.023, 0.06])
    p = g.sim_to_app(p_sim, gap)
    sim_off = wrap_diff(sim_module.focus_phases(p_sim, gap, "vortex"),
                        sim_module.focus_phases(p_sim, gap, "focus"))
    ours_off = wrap_diff(ph.vortex(p, charge=-1), ph.focus(p))
    np.testing.assert_allclose(
        wrap_diff(g.sim_raster_to_fpga(sim_off), ours_off + np.pi / 2), 0.0,
        atol=1e-4)  # sim uses an exact grid; ours has SonicSurface's float32 noise


def test_recipes_reject_bad_points():
    with pytest.raises(ValueError):
        ph.focus([0, 0.08])
