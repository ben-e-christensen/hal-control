"""GUI behaviour tests, run offscreen with a MockLink. No hardware."""

import os
import time
import xml.dom.minidom

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pg = pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtGui  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from hal_control import app as A  # noqa: E402
from hal_control import config as cfg  # noqa: E402
from hal_control import phases as ph  # noqa: E402
from hal_control.protocol import OFF_FRAME, ON_FRAME, parse_frame  # noqa: E402
from hal_control.serial_link import MockLink  # noqa: E402
from hal_control.trap import Output  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pg.setConfigOptions(imageAxisOrder="row-major")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(qapp, seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


@pytest.fixture
def win(qapp, tmp_path):
    links = []

    def factory(port):
        link = MockLink()
        links.append(link)
        return link

    w = A.MainWindow(cfg.Settings(gap_mm=120, trap_y_mm=80), tmp_path / "hal.json",
                     mock=True, link_factory=factory, show_field=False)
    w.alerts = []
    w._alert = lambda *a: w.alerts.append(a)
    w.links = links
    w.show()
    pump(qapp, 0.1)
    yield w
    w.close()
    pump(qapp, 0.05)


def wait_sent(qapp, link, n, timeout=1.0):
    end = time.time() + timeout
    while len(link.sent) < n and time.time() < end:
        pump(qapp, 0.01)
    link.wait_idle(timeout)
    return link.sent


def test_starts_disconnected_and_cannot_emit(win, qapp):
    assert "NOT CONNECTED" in win.banner.text()
    assert "NOT CONNECTED" in win.windowTitle()
    assert not win.on_btn.isEnabled() and not win.emit_btn.isEnabled()
    win.emit_trap()
    assert win.output is Output.OFF and win.alerts


def test_connect_sends_all_off_first(win, qapp):
    win.connect_link()
    link = win.links[-1]
    assert wait_sent(qapp, link, 1) == [OFF_FRAME]
    assert "MOCK" in win.banner.text() and "ALL OFF" in win.banner.text()


def test_trap_edits_while_off_only_preview(win, qapp):
    win.connect_link()
    link = win.links[-1]
    wait_sent(qapp, link, 1)
    win.x_spin.setValue(12.0)
    pump(qapp, 0.1)
    assert link.sent == [OFF_FRAME]


def test_emit_then_live_updates_and_esc(win, qapp):
    win.connect_link()
    link = win.links[-1]
    win.kind_combo.setCurrentText("vortex")
    win.emit_trap()
    wait_sent(qapp, link, 2)
    np.testing.assert_array_equal(parse_frame(link.sent[-1]),
                                  ph.discretize(ph.vortex(np.array([0, 0.08, 0]), 1)))
    for i in range(30):                      # fast drag: link coalesces, last one wins
        win._pick_top(i, -i, "drag")
    win._pick_top(30, -30, "finish")
    link.wait_idle(1.0)
    np.testing.assert_array_equal(parse_frame(link.sent[-1]),
                                  ph.discretize(ph.vortex(np.array([0.03, 0.08, -0.03]), 1)))
    assert "TRAP" in win.banner.text()

    QtWidgets.QApplication.sendEvent(
        win.x_spin, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                    Qt.KeyboardModifier.NoModifier))
    link.wait_idle(1.0)
    assert win.output is Output.OFF and link.sent[-1] == OFF_FRAME


def test_all_on_and_presets(win, qapp):
    win.connect_link()
    link = win.links[-1]
    win.all_on()
    link.wait_idle(1.0)
    assert link.sent[-1] == ON_FRAME
    win.preset_spin.setValue(25.0)
    win.preset_centre()
    assert (win.trap.x, win.trap.z) == (0.0, 0.0)
    assert win.trap.y == pytest.approx(0.095)
    assert win.above_spin.value() == pytest.approx(25.0)


def test_trap_clamped_between_array_and_plate(win):
    win._pick_side(0.0, 500.0, "click")
    assert win.trap.y == pytest.approx(0.119)
    win._pick_side(0.0, -20.0, "click")
    assert win.trap.y == pytest.approx(0.001)
    win.gap_spin.setValue(60.0)
    assert win.trap.y <= 0.059


def test_disconnect_and_close_send_off(win, qapp):
    win.connect_link()
    link = win.links[-1]
    win.all_on()
    link.wait_idle(1.0)
    win.disconnect_link()
    assert link.sent[-1] == OFF_FRAME and link.port_closed
    assert "NOT CONNECTED" in win.banner.text()
    win.connect_link()
    link2 = win.links[-1]
    win.all_on()
    link2.wait_idle(1.0)
    win.close()
    assert link2.sent[-1] == OFF_FRAME


def test_link_error_shows_unknown_state(win, qapp):
    win.connect_link()
    link = win.links[-1]
    link.wait_idle(1.0)
    link.fail_writes = {link.write_calls}     # the next write fails
    win.all_on()
    end = time.time() + 2
    while "LINK ERROR" not in win.banner.text() and time.time() < end:
        pump(qapp, 0.02)
    assert "LINK ERROR" in win.banner.text() and "UNKNOWN" in win.banner.text()
    assert win.link is None and win.alerts


def test_settings_persist(win, qapp, tmp_path):
    win.gap_spin.setValue(100.0)
    win.kind_combo.setCurrentText("twin")
    win.angle_spin.setValue(45)
    win.save_settings()
    s = cfg.load(tmp_path / "hal.json")
    assert s.gap_mm == 100.0 and s.trap_kind == "twin" and s.trap_angle_deg == 45
    assert s.port != A.MOCK_PORT


def test_field_view_and_exports(win, qapp, tmp_path):
    win.show_field_cb.setChecked(True)
    win._request_field("fine")
    end = time.time() + 5
    while getattr(win, "_display", None) is None and time.time() < end:
        pump(qapp, 0.05)
    assert win._display["res"].req.quality == "fine"
    for which in ("side", "top"):
        fn = tmp_path / f"{which}.svg"
        win.write_svg(which, fn)
        xml.dom.minidom.parse(str(fn))
    win.write_csv(tmp_path / "p.csv")
    assert "preview" in (tmp_path / "p.csv").read_text().splitlines()[0]
