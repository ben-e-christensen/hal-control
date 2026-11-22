"""Fixtures that load the read-only reference code in refs/ without modifying it."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "refs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSerial:
    """Captures what SonicSurface.py would write to the port."""

    def __init__(self):
        self.data = bytearray()

    def write(self, b):
        self.data += bytes(b)

    def close(self):
        pass


@pytest.fixture(scope="session")
def ss_module():
    return _load("ref_sonicsurface", REFS / "SonicSurface.py")


@pytest.fixture
def ss(ss_module):
    """A SonicSurface instance whose serial port is a FakeSerial."""
    s = ss_module.SonicSurface()
    s.serialConn = FakeSerial()
    return s


@pytest.fixture(scope="session")
def sim_module():
    """refs/hal_lev_sim.py, importable without matplotlib (only its numpy parts are used)."""
    try:
        import matplotlib.patheffects  # noqa: F401
    except ImportError:
        mpl = types.ModuleType("matplotlib")
        pe = types.ModuleType("matplotlib.patheffects")
        pe.withStroke = lambda **kw: None
        mpl.patheffects = pe
        sys.modules.setdefault("matplotlib", mpl)
        sys.modules.setdefault("matplotlib.patheffects", pe)
    return _load("ref_hal_lev_sim", REFS / "hal_lev_sim.py")


@pytest.fixture(scope="session")
def ultraino_transducers():
    from ultraino_xml import transducers
    return transducers(REFS / "emptyboard.xml.gz")
