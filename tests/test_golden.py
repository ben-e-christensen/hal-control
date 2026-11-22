"""Golden test: our frames vs what refs/SonicSurface.py writes to the port.

Differences accounted for:
  * SonicSurface works in a frame offset by +2 mm in Z (geometry.SS_ORIGIN_OFFSET).
  * SonicSurface truncates phases to codes; we round. So every code is either
    equal to SonicSurface's or exactly one step higher (mod 32), and equals
    round-half-up of SonicSurface's continuous phase.
  * We prefix each frame with 192 (enable primary board).
"""

import itertools

import numpy as np
import pytest

from hal_control import geometry as g
from hal_control import phases as ph
from hal_control.protocol import OFF_FRAME, ON_FRAME, frame_from_phases

GRID = [np.array(p) / 1000 for p in itertools.product((-40, 0, 25), (20, 60, 140), (-30, 0, 35))]
RANDOM = list(np.random.default_rng(1).uniform([-0.06, 0.01, -0.06], [0.06, 0.16, 0.06], (30, 3)))


def ss_frame(ss, p):
    ss.serialConn.data.clear()
    s = g.app_to_ss(p)
    ss.focusAtPos(s[0], s[1], s[2])
    return bytes(ss.serialConn.data)


@pytest.mark.parametrize("p", GRID + RANDOM)
def test_focus_frame_matches_sonicsurface(ss, p):
    ref = ss_frame(ss, p)
    ours = frame_from_phases(ph.focus(p))
    # framing: SonicSurface sends 254 + 256 + 253; we send 192 first
    assert len(ref) == 258 and ref[0] == 254 and ref[-1] == 253
    assert ours[0] == 192 and ours[1:2] == ref[:1] and ours[-1:] == ref[-1:]

    ref_codes = np.frombuffer(ref[1:-1], np.uint8).astype(int)
    our_codes = np.frombuffer(ours[2:-1], np.uint8).astype(int)

    # rounding vs truncation: equal or one step up
    assert set(np.unique((our_codes - ref_codes) % 32)) <= {0, 1}

    # exact: our code = round-half-up of SonicSurface's continuous phase
    cont = ss.phasesForFocus(g.app_to_ss(p)) % (2 * np.pi) * 32 / (2 * np.pi)
    expected = np.floor(cont + 0.5).astype(int) % 32
    tie = np.abs(cont - np.floor(cont) - 0.5) < 1e-9  # float noise can flip exact ties
    np.testing.assert_array_equal(our_codes[~tie], expected[~tie])


def test_on_off_frames_match_sonicsurface(ss):
    for on, ours in ((True, ON_FRAME), (False, OFF_FRAME)):
        ss.serialConn.data.clear()
        ss.switchOnOrOff(on)
        assert ours[1:] == bytes(ss.serialConn.data)
