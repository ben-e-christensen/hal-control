import io

import numpy as np
import pytest

from hal_control import cli
from hal_control import phases as ph


def run(*argv):
    out = io.StringIO()
    rc = cli.main(["--mock", *argv], out=out)
    return rc, out.getvalue()


def printed_codes(text):
    lines = text.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("codes,"))
    return np.array([[int(v) for v in l.split()] for l in lines[i + 1:i + 17]]).ravel()


def test_off_on():
    rc, out = run("off")
    assert rc == 0 and (printed_codes(out) == 32).all()
    rc, out = run("on")
    assert rc == 0 and (printed_codes(out) == 0).all()


def test_focus_twin_vortex_codes():
    p = np.array([5, 80, -10]) / 1000
    _, out = run("focus", "5", "80", "-10")
    np.testing.assert_array_equal(printed_codes(out), ph.discretize(ph.focus(p)))
    _, out = run("twin", "5", "80", "-10", "--angle", "90")
    np.testing.assert_array_equal(printed_codes(out), ph.discretize(ph.twin(p, np.pi / 2)))
    _, out = run("vortex", "5", "80", "-10", "--charge", "-2")
    np.testing.assert_array_equal(printed_codes(out), ph.discretize(ph.vortex(p, -2)))


def test_gap_prints_height_above_plate():
    _, out = run("--gap", "120", "focus", "0", "80", "0")
    assert "40 mm above plate" in out


def test_rejects_non_positive_y():
    with pytest.raises(SystemExit):
        run("focus", "0", "0", "0")
