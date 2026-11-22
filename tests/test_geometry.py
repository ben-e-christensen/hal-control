import ast
import re

import numpy as np

from hal_control import geometry as g
from conftest import REFS


def test_shape_and_plane():
    assert g.EMITTER_POS.shape == (256, 3)
    assert np.all(g.EMITTER_POS[:, 1] == 0.0)


def test_grid_is_16x16_at_pitch_and_centred():
    for axis in (0, 2):
        levels = np.unique(np.round(g.EMITTER_POS[:, axis], 6))
        assert len(levels) == 16
        np.testing.assert_allclose(np.diff(levels), g.PITCH, atol=1e-6)
        np.testing.assert_allclose(levels.mean(), 0.0, atol=1e-6)
    cells = set(zip(g.GRID_ROW.tolist(), g.GRID_COL.tolist()))
    assert len(cells) == 256


def test_verbatim_copy_of_sonicsurface_positions(ss_module):
    ref = np.array(ss_module.SonicSurface.EMITTERS_POS, dtype=float).reshape(-1, 3)
    np.testing.assert_array_equal(g.SS_EMITTER_POS, ref)


def test_sonicsurface_frame_is_2mm_off_centre_in_z():
    np.testing.assert_allclose(g.SS_EMITTER_POS.mean(axis=0), [0, 0, 0.002], atol=1e-6)
    np.testing.assert_allclose(g.EMITTER_POS, g.ss_to_app(g.SS_EMITTER_POS))
    np.testing.assert_allclose(g.app_to_ss(g.ss_to_app([1.0, 2.0, 3.0])), [1, 2, 3])


def test_channel_order_matches_ultraino_emptyboard(ultraino_transducers):
    ts = ultraino_transducers
    assert [t["order"] for t in ts] == list(range(256))
    ultraino = np.array([t["pos"] for t in ts])
    np.testing.assert_allclose(g.EMITTER_POS, ultraino, atol=1e-6)


def test_sonicsurface_raster_permutation_agrees_with_geometry():
    """SonicSurface.sendPattern's EMITTERS_ORDER: raster i -> FPGA channel ORDER[i],
    raster row 0 at max Z, columns increasing in X."""
    src = (REFS / "SonicSurface.py").read_text()
    order = ast.literal_eval(re.search(r"EMITTERS_ORDER = (\[.*?\])", src).group(1))
    assert sorted(order) == list(range(256))
    for i, ch in enumerate(order):
        assert (15 - g.GRID_ROW[ch]) * 16 + g.GRID_COL[ch] == i


def test_sim_raster_mapping(sim_module):
    sim_x = g.sim_raster_to_fpga(sim_module.EMX)
    sim_y = g.sim_raster_to_fpga(sim_module.EMY)
    np.testing.assert_allclose(sim_x, g.EMITTER_POS[:, 0], atol=1e-6)
    np.testing.assert_allclose(sim_y, g.EMITTER_POS[:, 2], atol=1e-6)
    v = np.arange(256)
    np.testing.assert_array_equal(g.fpga_to_sim_raster(g.sim_raster_to_fpga(v)), v)
    np.testing.assert_array_equal(g.sim_raster_to_fpga(g.fpga_to_sim_raster(v)), v)


def test_sim_axis_conversion():
    gap = 0.10
    np.testing.assert_allclose(g.sim_to_app([0.001, 0.002, 0.003], gap), [0.001, 0.097, 0.002])
    # board face (sim z = H) -> Y = 0; plate (sim z = 0) -> Y = gap
    assert g.sim_to_app([0, 0, gap], gap)[1] == 0.0
    assert g.sim_to_app([0, 0, 0], gap)[1] == gap
    pts = np.random.default_rng(0).uniform(-0.05, 0.05, (20, 3))
    np.testing.assert_allclose(g.app_to_sim(g.sim_to_app(pts, gap), gap), pts)
    np.testing.assert_allclose(g.sim_to_app(g.app_to_sim(pts, gap), gap), pts)
