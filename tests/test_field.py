import numpy as np
import pytest

from hal_control import field as f
from hal_control import phases as ph
from hal_control.geometry import WAVELENGTH

GAP = 0.12
TRAP = np.array([0.01, 0.07, -0.005])


def sources(rad, **kw):
    return f.make_sources(f.model_phases(ph.discretize(rad)), f.FieldModel(gap=GAP, **kw))


def rand_pts(n=300, seed=0):
    return np.random.default_rng(seed).uniform([-0.09, 0.002, -0.09], [0.09, GAP - 0.002, 0.09],
                                               (n, 3))


@pytest.mark.parametrize("rad", [ph.focus(TRAP), ph.vortex(TRAP, 2), ph.all_on()])
def test_fast_kernel_matches_reference(rad):
    src = sources(rad)
    pts = rand_pts()
    p, g = f.pressure(src, pts, want_grad=True)
    pr, gr = f.pressure_reference(src, pts, want_grad=True)
    assert np.abs(p - pr).max() < 2e-3 * np.abs(pr).max()
    assert np.abs(g - gr).max() < 2e-3 * np.abs(gr).max()


def test_off_codes_are_not_sources():
    rad = ph.focus(TRAP)
    rad[:100] = np.nan
    src = sources(rad, reflector=False)
    assert len(src.pos) == 156
    assert len(sources(ph.all_off()).pos) == 0
    p, _ = f.pressure(sources(ph.all_off()), rand_pts(10))
    assert np.all(p == 0)


def test_image_sources_are_mirrored_across_plate():
    src = sources(ph.focus(TRAP), refl_coeff=0.8)
    real, img = src.pos[:256], src.pos[256:]
    np.testing.assert_allclose(img[:, [0, 2]], real[:, [0, 2]])
    np.testing.assert_allclose(img[:, 1], 2 * GAP - real[:, 1])
    np.testing.assert_allclose(src.amp[256:], 0.8 * src.amp[:256])
    assert np.all(src.normal_y[:256] == 1) and np.all(src.normal_y[256:] == -1)


def test_rigid_plate_boundary_conditions():
    """R = 1: no normal velocity at the plate (dp/dY = 0) and pressure doubling."""
    pts = rand_pts(200)
    pts[:, 1] = GAP
    full = sources(ph.vortex(TRAP), refl_coeff=1.0)
    free = sources(ph.vortex(TRAP), reflector=False)
    p, g = f.pressure_reference(full, pts, want_grad=True)
    p0, _ = f.pressure_reference(free, pts)
    assert np.abs(g[:, 1]).max() < 1e-9 * np.abs(g).max()
    np.testing.assert_allclose(p, 2 * p0, rtol=1e-9)


def test_reflection_coefficient_is_linear_and_zero_is_free_field():
    pts = rand_pts(100)
    p_free = f.pressure_reference(sources(ph.focus(TRAP), reflector=False), pts)[0]
    p_r0 = f.pressure_reference(sources(ph.focus(TRAP), refl_coeff=0.0), pts)[0]
    p_r1 = f.pressure_reference(sources(ph.focus(TRAP), refl_coeff=1.0), pts)[0]
    p_r5 = f.pressure_reference(sources(ph.focus(TRAP), refl_coeff=0.5), pts)[0]
    np.testing.assert_allclose(p_r0, p_free)
    np.testing.assert_allclose(p_r5, p_free + 0.5 * (p_r1 - p_free), rtol=1e-9, atol=1e-9)


def test_free_field_focus_is_the_brightest_point():
    src = sources(ph.focus(TRAP), reflector=False)
    sl = f.side_slice(src, GAP, TRAP[2], 120, 80)
    iv, ih = np.unravel_index(np.argmax(sl.values), sl.values.shape)
    peak = np.array([sl.h[ih], sl.v[iv]])
    assert np.linalg.norm(peak - TRAP[:2]) < 4e-3


def test_slices_have_expected_shapes_and_axes():
    src = sources(ph.focus(TRAP))
    side = f.side_slice(src, GAP, 0.0, 30, 20)
    top = f.top_slice(src, 0.07, 25, 15, what="gorkov")
    assert side.values.shape == (20, 30) and top.values.shape == (15, 25)
    assert 0 < side.v[0] < side.v[-1] < GAP
    assert np.all(np.isfinite(top.values))


def test_standing_wave_nodes_half_a_wavelength_apart():
    nodes = f.pressure_nodes(sources(ph.all_on()), GAP, 0.0, 0.0)
    assert len(nodes) > 15
    np.testing.assert_allclose(np.median(np.diff(nodes)), WAVELENGTH / 2, rtol=0.1)
    assert np.all((nodes > 0) & (nodes < GAP))


def test_vortex_core_gives_no_spurious_nodes():
    nodes = f.pressure_nodes(sources(ph.vortex(np.array([0, 0.08, 0]))), GAP, 0.0, 0.0)
    assert np.diff(nodes).min() > 0.25 * WAVELENGTH


def test_gorkov_minima_sit_near_pressure_nodes_in_standing_wave():
    src = sources(ph.all_on())
    y = np.linspace(0.03, 0.09, 1200)
    pts = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])
    U = f.gorkov(src, pts)
    mins = y[1:-1][(U[1:-1] < U[:-2]) & (U[1:-1] < U[2:])]
    nodes = f.pressure_nodes(src, GAP, 0.0, 0.0)
    nodes = nodes[(nodes > 0.035) & (nodes < 0.085)]
    assert len(nodes) >= 8
    for n in nodes:
        assert np.abs(mins - n).min() < 1e-3
