"""HAL Control GUI (PySide6 + pyqtgraph).

Safety model
------------
* The app starts NOT CONNECTED and never emits on launch.
* Connect immediately sends ALL OFF, so the board matches the app's state
  (the board's own power-on default is all on).
* Output is one of OFF / ON / TRAP. Editing the trap while OFF only changes
  the preview; EMIT TRAP starts sending it, after which every change is sent
  live (the link coalesces to the newest frame at <= ~89 frames/s).
* ALL OFF (button or Esc anywhere) always works. Disconnect, window close,
  Ctrl+C in the terminal and link errors all end with ALL OFF.
"""

import math
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, QTimer, Signal

from . import config as cfg
from . import field as fld
from . import export as export_mod
from .export import phases_csv
from .geometry import EMITTER_POS, N_SIDE, PITCH
from .protocol import FRAME_TIME, MAX_FPS, OFF_FRAME
from .serial_link import MockLink, SerialLink
from .trap import KINDS, Output, Trap, mm_above_plate, output_codes

MOCK_PORT = "mock"
HALF_VIEW_MM = 95.0                                   # field view half-width
ARRAY_HALF_MM = N_SIDE * PITCH / 2 * 1e3              # 84 mm (outer edge of the grid)
Y_MARGIN_MM = 1.0                                     # keep the trap off the array and the plate

GRID = {  # (h, v) samples; side view v is scaled by the gap
    "coarse": {"side": 64, "top": 64},
    "fine": {"side": 190, "top": 170},
}


# =============================================================================
# Port helpers
# =============================================================================
def list_ports(include_mock=False):
    """USB/ACM-style serial ports first; legacy /dev/ttyS* hidden (there are dozens)."""
    try:
        from serial.tools import list_ports as lp
        devs = [p.device for p in lp.comports()]
    except Exception:  # noqa: BLE001
        devs = []
    devs = sorted(d for d in devs if not re.fullmatch(r"/dev/ttyS\d+", d))
    if include_mock:
        devs.append(MOCK_PORT)
    return devs


def default_link_factory(port):
    if port == MOCK_PORT:
        return MockLink(write_time=FRAME_TIME)  # behaves like the real wire
    return SerialLink(port)


# =============================================================================
# Field computation thread (latest request wins)
# =============================================================================
@dataclass(frozen=True)
class FieldRequest:
    seq: int
    codes: np.ndarray
    model: fld.FieldModel
    trap: Trap
    view: str          # "amplitude" | "gorkov"
    quality: str       # "coarse" | "fine"


@dataclass(frozen=True)
class FieldResult:
    req: FieldRequest
    side: fld.Slice
    top: fld.Slice
    nodes: np.ndarray  # Y of pressure nodes along the trap's vertical line (m)
    seconds: float


def compute_field(req):
    t0 = time.perf_counter()
    src = fld.make_sources(fld.model_phases(req.codes), req.model)
    g = GRID[req.quality]
    gap = req.model.gap
    n_side_v = max(16, int(round(g["side"] * gap / (2 * HALF_VIEW_MM / 1e3))))
    side = fld.side_slice(src, gap, req.trap.z, g["side"], n_side_v,
                          half_width=HALF_VIEW_MM / 1e3, what=req.view)
    top = fld.top_slice(src, req.trap.y, g["top"], g["top"],
                        half_width=HALF_VIEW_MM / 1e3, what=req.view)
    nodes = fld.pressure_nodes(src, gap, req.trap.x, req.trap.z, n=500)
    return FieldResult(req, side, top, nodes, time.perf_counter() - t0)


class FieldWorker(QThread):
    done = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending = None
        self._stop = False

    def request(self, req):
        with self._cond:
            self._pending = req
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.wait(3000)

    def run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                req, self._pending = self._pending, None
            try:
                res = compute_field(req)
            except Exception:  # noqa: BLE001 - never kill the worker
                traceback.print_exc()
                continue
            self.done.emit(res)


# =============================================================================
# Plot widgets: fixed views where click / drag anywhere moves the trap
# =============================================================================
class PickViewBox(pg.ViewBox):
    """ViewBox without pan/zoom; left click or drag reports data coordinates."""

    def __init__(self, on_pick):
        super().__init__(enableMenu=False)
        self.on_pick = on_pick          # on_pick(x, y, phase) phase: click|drag|finish
        self.setMouseEnabled(False, False)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            self.on_pick(p.x(), p.y(), "click")
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        p = self.mapSceneToView(ev.scenePos())
        self.on_pick(p.x(), p.y(), "finish" if ev.isFinish() else "drag")

    def wheelEvent(self, ev, axis=None):
        ev.ignore()


def make_plot(on_pick, title, xlabel, ylabel):
    vb = PickViewBox(on_pick)
    w = pg.PlotWidget(viewBox=vb)
    pi = w.getPlotItem()
    pi.setTitle(title)
    pi.setLabel("bottom", xlabel)
    pi.setLabel("left", ylabel)
    pi.hideButtons()
    vb.setAspectLocked(True)
    w.setBackground("w")
    return w


def place_image(img, sl):
    """Position an ImageItem (row-major) so pixel centres sit on the slice grid (mm)."""
    h, v = sl.h * 1e3, sl.v * 1e3
    dh = (h[-1] - h[0]) / max(1, len(h) - 1)
    dv = (v[-1] - v[0]) / max(1, len(v) - 1)
    img.setRect(QtCore.QRectF(h[0] - dh / 2, v[0] - dv / 2, h[-1] - h[0] + dh, v[-1] - v[0] + dv))


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    link_error = Signal(str)

    def __init__(self, settings, settings_path=cfg.DEFAULT_PATH, mock=False,
                 link_factory=default_link_factory, show_field=None):
        super().__init__()
        self.settings = settings
        self.settings_path = settings_path
        self.mock = mock
        self.link_factory = link_factory
        self.link = None
        self.link_port = None
        self.link_err = None
        self.output = Output.OFF
        self.gap = settings.gap_mm / 1e3
        self.trap = self._clamp(settings.trap())
        self.dragging = False
        self._syncing = False
        self._field_seq = 0
        self._last_result = None
        self._nodes = np.array([])

        self.link_error.connect(self._on_link_error)

        self.worker = FieldWorker(self)
        self.worker.done.connect(self._on_field)
        self.worker.start()

        self._fine_timer = QTimer(self, singleShot=True, interval=180)
        self._fine_timer.timeout.connect(lambda: self._request_field("fine"))
        self._save_timer = QTimer(self, singleShot=True, interval=1000)
        self._save_timer.timeout.connect(self.save_settings)
        self._status_timer = QTimer(self, interval=250)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

        self._build_ui()
        if show_field is not None:
            self.show_field_cb.setChecked(show_field)
        self._sync_widgets()
        self._update_banner()
        self._request_field("fine")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        s = self.settings
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # --- banner: connection state + output state, impossible to miss ---
        self.banner = QtWidgets.QLabel()
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setMinimumHeight(54)
        self.banner.setWordWrap(True)
        f = self.banner.font()
        f.setPointSize(f.pointSize() + 4)
        f.setBold(True)
        self.banner.setFont(f)
        root.addWidget(self.banner)

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, 1)

        # --- left: controls ---------------------------------------------------
        left = QtWidgets.QWidget()
        left.setFixedWidth(360)
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setFixedWidth(380)
        body.addWidget(scroll)

        # Connection
        g = QtWidgets.QGroupBox("Connection")
        gl = QtWidgets.QGridLayout(g)
        self.port_combo = QtWidgets.QComboBox(editable=True)
        self._fill_ports(s.port if not self.mock else MOCK_PORT)
        refresh = QtWidgets.QPushButton("Rescan")
        refresh.clicked.connect(lambda: self._fill_ports(self.port_combo.currentText()))
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.light = QtWidgets.QLabel()
        self.light.setFixedSize(22, 22)
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setStyleSheet("font-family: monospace;")
        self.stats_label.setWordWrap(True)
        gl.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        gl.addWidget(self.port_combo, 0, 1)
        gl.addWidget(refresh, 0, 2)
        gl.addWidget(self.light, 1, 0)
        gl.addWidget(self.connect_btn, 1, 1, 1, 2)
        gl.addWidget(self.stats_label, 2, 0, 1, 3)
        lv.addWidget(g)

        # Output
        g = QtWidgets.QGroupBox("Output")
        gl = QtWidgets.QVBoxLayout(g)
        self.off_btn = QtWidgets.QPushButton("ALL OFF   (Esc)")
        self.off_btn.setMinimumHeight(64)
        self.off_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font-size:20px;font-weight:bold;"
            "border-radius:6px} QPushButton:pressed{background:#8e0000}")
        self.off_btn.clicked.connect(self.all_off)
        row = QtWidgets.QHBoxLayout()
        self.on_btn = QtWidgets.QPushButton("ALL ON\n(standing wave)")
        self.on_btn.clicked.connect(self.all_on)
        self.emit_btn = QtWidgets.QPushButton("EMIT TRAP")
        self.emit_btn.setStyleSheet("QPushButton{font-weight:bold}")
        self.emit_btn.clicked.connect(self.emit_trap)
        for b in (self.on_btn, self.emit_btn):
            b.setMinimumHeight(44)
            row.addWidget(b)
        gl.addWidget(self.off_btn)
        gl.addLayout(row)
        lv.addWidget(g)

        # Trap
        g = QtWidgets.QGroupBox("Trap (mm, app frame: Y = out of the array face)")
        gl = QtWidgets.QGridLayout(g)
        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.addItems(KINDS)
        self.kind_combo.currentTextChanged.connect(self._on_kind)

        def dspin(lo, hi, step=0.5, dec=1, suffix=" mm"):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setKeyboardTracking(False)
            return sp

        self.x_spin = dspin(-HALF_VIEW_MM, HALF_VIEW_MM)
        self.y_spin = dspin(Y_MARGIN_MM, 1000)
        self.z_spin = dspin(-HALF_VIEW_MM, HALF_VIEW_MM)
        self.above_spin = dspin(Y_MARGIN_MM, 1000)
        self.gap_spin = dspin(10, 400, 1.0)
        self.gap_spin.setValue(s.gap_mm)
        self.angle_spin = dspin(-360, 360, 15, 0, " deg")
        self.charge_spin = QtWidgets.QSpinBox()
        self.charge_spin.setRange(-8, 8)
        self.charge_spin.setKeyboardTracking(False)
        for sp in (self.x_spin, self.y_spin, self.z_spin):
            sp.valueChanged.connect(self._on_xyz_spin)
        self.above_spin.valueChanged.connect(self._on_above_spin)
        self.gap_spin.valueChanged.connect(self._on_gap)
        self.angle_spin.valueChanged.connect(
            lambda v: self._syncing or self.set_trap(self.trap.moved(angle_deg=v)))
        self.charge_spin.valueChanged.connect(
            lambda v: self._syncing or self.set_trap(self.trap.moved(charge=v)))

        rows = [("Type", self.kind_combo), ("X", self.x_spin), ("Y (from array)", self.y_spin),
                ("Z", self.z_spin), ("Above plate", self.above_spin),
                ("Gap (array to plate)", self.gap_spin), ("Twin angle", self.angle_spin),
                ("Vortex charge", self.charge_spin)]
        for i, (lab, w) in enumerate(rows):
            gl.addWidget(QtWidgets.QLabel(lab), i, 0)
            gl.addWidget(w, i, 1)
        self.snap_btn = QtWidgets.QPushButton("Snap Y to nearest pressure node")
        self.snap_btn.clicked.connect(self.snap_to_node)
        gl.addWidget(self.snap_btn, len(rows), 0, 1, 2)
        self.nodes_label = QtWidgets.QLabel()
        self.nodes_label.setWordWrap(True)
        gl.addWidget(self.nodes_label, len(rows) + 1, 0, 1, 2)
        lv.addWidget(g)

        # Presets
        g = QtWidgets.QGroupBox("Presets")
        gl = QtWidgets.QGridLayout(g)
        self.preset_spin = dspin(Y_MARGIN_MM, 400)
        self.preset_spin.setValue(s.preset_above_plate_mm)
        self.preset_spin.valueChanged.connect(lambda _: self._schedule_save())
        b = QtWidgets.QPushButton("Centre, N mm above plate")
        b.clicked.connect(self.preset_centre)
        gl.addWidget(QtWidgets.QLabel("N"), 0, 0)
        gl.addWidget(self.preset_spin, 0, 1)
        gl.addWidget(b, 1, 0, 1, 2)
        b = QtWidgets.QPushButton("Plain standing wave (all on, phase 0)")
        b.clicked.connect(self.all_on)
        gl.addWidget(b, 2, 0, 1, 2)
        lv.addWidget(g)

        # Field view
        g = QtWidgets.QGroupBox("Field view (model)")
        gl = QtWidgets.QGridLayout(g)
        self.show_field_cb = QtWidgets.QCheckBox("Show field")
        self.show_field_cb.setChecked(s.show_field)
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem("Amplitude |p|", "amplitude")
        self.view_combo.addItem("Gor'kov potential U", "gorkov")
        self.view_combo.setCurrentIndex(0 if s.field_view == "amplitude" else 1)
        self.refl_cb = QtWidgets.QCheckBox("Reflector (image sources)")
        self.refl_cb.setChecked(s.reflector)
        self.refl_spin = dspin(0.0, 1.0, 0.05, 2, "")
        self.refl_spin.setValue(s.refl_coeff)
        self.vpp_spin = dspin(1.0, 30.0, 1.0, 1, " Vpp")
        self.vpp_spin.setValue(s.model_vpp)
        self.vnote = QtWidgets.QLineEdit(s.voltage_note)
        self.vnote.setPlaceholderText("drive voltage note (saved)")
        for w_ in (self.show_field_cb, self.refl_cb):
            w_.toggled.connect(self._on_field_settings)
        self.view_combo.currentIndexChanged.connect(self._on_field_settings)
        for w_ in (self.refl_spin, self.vpp_spin):
            w_.valueChanged.connect(self._on_field_settings)
        self.vnote.editingFinished.connect(self._schedule_save)
        self.field_label = QtWidgets.QLabel()
        self.field_label.setWordWrap(True)
        gl.addWidget(self.show_field_cb, 0, 0)
        gl.addWidget(self.view_combo, 0, 1)
        gl.addWidget(self.refl_cb, 1, 0)
        gl.addWidget(self.refl_spin, 1, 1)
        gl.addWidget(QtWidgets.QLabel("Model drive"), 2, 0)
        gl.addWidget(self.vpp_spin, 2, 1)
        gl.addWidget(self.vnote, 3, 0, 1, 2)
        gl.addWidget(self.field_label, 4, 0, 1, 2)
        lv.addWidget(g)

        # Export
        g = QtWidgets.QGroupBox("Export")
        gl = QtWidgets.QHBoxLayout(g)
        for text, fn in (("Side SVG", lambda: self.export_svg("side")),
                         ("Top SVG", lambda: self.export_svg("top")),
                         ("Phases CSV", self.export_csv)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            gl.addWidget(b)
        lv.addWidget(g)
        lv.addStretch(1)

        # --- right: plots -------------------------------------------------------
        right = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(right, 1)

        self.top_plot = make_plot(self._pick_top, "", "X (mm)", "Z (mm)")
        side_box = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(side_box)
        sh.setContentsMargins(0, 0, 0, 0)
        self.side_plot = make_plot(self._pick_side, "", "X (mm)", "Y (mm from array face)")
        self.side_plot.getPlotItem().getViewBox().invertY(True)   # board on top, plate below
        sliders = QtWidgets.QVBoxLayout()
        sliders.addWidget(QtWidgets.QLabel("array"), 0, Qt.AlignmentFlag.AlignHCenter)
        self.y_slider = QtWidgets.QSlider(Qt.Orientation.Vertical)
        self.y_slider.setInvertedAppearance(True)                  # top = near the array
        self.y_slider.valueChanged.connect(self._on_slider)
        self.y_slider.sliderPressed.connect(lambda: self._set_dragging(True))
        self.y_slider.sliderReleased.connect(lambda: self._set_dragging(False))
        sliders.addWidget(self.y_slider, 1, Qt.AlignmentFlag.AlignHCenter)
        sliders.addWidget(QtWidgets.QLabel("plate"), 0, Qt.AlignmentFlag.AlignHCenter)
        sh.addWidget(self.side_plot, 1)
        sh.addLayout(sliders)
        right.addWidget(self.top_plot)
        right.addWidget(side_box)

        # images + colour bar
        self.top_img = pg.ImageItem()
        self.side_img = pg.ImageItem()
        self.top_plot.addItem(self.top_img)
        self.side_plot.addItem(self.side_img)
        self.cmaps = {"amplitude": pg.colormap.get("inferno"),
                      "gorkov": pg.colormap.get("viridis")}
        self.cbar = pg.ColorBarItem(colorMap=self.cmaps["amplitude"], interactive=False,
                                    width=14, colorMapMenu=False)
        self.cbar.setImageItem([self.side_img, self.top_img],
                               insert_in=self.side_plot.getPlotItem())

        # static overlays: array outline + emitter dots (top), array face + plate (side)
        a = ARRAY_HALF_MM
        self.top_plot.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a],
                           pen=pg.mkPen("#4fc3f7", width=1.5, style=Qt.PenStyle.DashLine))
        self.top_plot.addItem(pg.ScatterPlotItem(
            EMITTER_POS[:, 0] * 1e3, EMITTER_POS[:, 2] * 1e3, size=3,
            pen=None, brush=pg.mkBrush(180, 180, 180, 110)))
        self.side_plot.plot([-a, a], [0, 0], pen=pg.mkPen("#4fc3f7", width=5))
        self.plate_line = self.side_plot.plot([-HALF_VIEW_MM, HALF_VIEW_MM], [0, 0],
                                              pen=pg.mkPen("#9e9e9e", width=5))
        self.node_marks = pg.ScatterPlotItem(symbol="t1", size=9, pen=pg.mkPen("k"),
                                             brush=pg.mkBrush("#00e5ff"))
        self.side_plot.addItem(self.node_marks)
        tpen = pg.mkPen("#00e676", width=2)
        self.top_target = pg.TargetItem(size=18, symbol="crosshair", movable=False, pen=tpen)
        self.side_target = pg.TargetItem(size=18, symbol="crosshair", movable=False, pen=tpen)
        self.top_plot.addItem(self.top_target)
        self.side_plot.addItem(self.side_target)
        self.top_plot.setRange(xRange=(-HALF_VIEW_MM, HALF_VIEW_MM),
                               yRange=(-HALF_VIEW_MM, HALF_VIEW_MM), padding=0.02)

        self.setCentralWidget(central)
        self.resize(1500, 860)

        # Esc anywhere -> ALL OFF (application-level filter, beats widget key handling)
        self._esc = _EscFilter(self.all_off)
        QtWidgets.QApplication.instance().installEventFilter(self._esc)

    def _fill_ports(self, select):
        ports = list_ports(include_mock=self.mock)
        if select and select not in ports:
            ports.insert(0, select)
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        self.port_combo.setCurrentText(select or (ports[0] if ports else ""))

    # ------------------------------------------------------------ state sync
    def _clamp(self, t):
        h = HALF_VIEW_MM / 1e3
        y_lo, y_hi = Y_MARGIN_MM / 1e3, self.gap - Y_MARGIN_MM / 1e3
        return t.moved(x=float(np.clip(t.x, -h, h)), z=float(np.clip(t.z, -h, h)),
                       y=float(np.clip(t.y, y_lo, max(y_lo, y_hi))))

    def _sync_widgets(self):
        """Push self.trap / self.gap into every widget without feedback loops."""
        self._syncing = True
        try:
            t, gap_mm = self.trap, self.gap * 1e3
            self.kind_combo.setCurrentText(t.kind)
            self.x_spin.setValue(t.x * 1e3)
            self.y_spin.setMaximum(gap_mm - Y_MARGIN_MM)
            self.y_spin.setValue(t.y * 1e3)
            self.z_spin.setValue(t.z * 1e3)
            self.above_spin.setMaximum(gap_mm - Y_MARGIN_MM)
            self.above_spin.setValue(mm_above_plate(t.y, self.gap))
            self.angle_spin.setValue(t.angle_deg)
            self.charge_spin.setValue(t.charge)
            self.angle_spin.setEnabled(t.kind == "twin")
            self.charge_spin.setEnabled(t.kind == "vortex")
            self.y_slider.setRange(int(Y_MARGIN_MM * 10), int((gap_mm - Y_MARGIN_MM) * 10))
            self.y_slider.setValue(int(round(t.y * 1e4)))
            self.top_target.setPos(t.x * 1e3, t.z * 1e3)
            self.side_target.setPos(t.x * 1e3, t.y * 1e3)
            self.plate_line.setData([-HALF_VIEW_MM, HALF_VIEW_MM], [gap_mm, gap_mm])
            self.side_plot.setRange(xRange=(-HALF_VIEW_MM, HALF_VIEW_MM),
                                    yRange=(-4, gap_mm + 4), padding=0.02)
            self.top_plot.getPlotItem().setTitle(
                f"Top view (XZ) at Y = {t.y * 1e3:.1f} mm  "
                f"({mm_above_plate(t.y, self.gap):.1f} mm above plate)")
            self.side_plot.getPlotItem().setTitle(f"Side view (XY) at Z = {t.z * 1e3:.1f} mm")
        finally:
            self._syncing = False

    def set_trap(self, trap, dragging=None):
        """Single entry point for every trap change (spinboxes, pad, slider, presets)."""
        if dragging is not None:
            self.dragging = dragging
        self.trap = self._clamp(trap)
        self._sync_widgets()
        if self.output is Output.TRAP:
            self._send()
        self._update_banner()
        self._field_changed()
        self._schedule_save()

    def _set_dragging(self, on):
        self.dragging = on
        if not on:
            self._request_field("fine")

    # widget callbacks
    def _on_kind(self, kind):
        if not self._syncing:
            self.set_trap(self.trap.moved(kind=kind))

    def _on_xyz_spin(self, _):
        if not self._syncing:
            self.set_trap(self.trap.moved(x=self.x_spin.value() / 1e3,
                                          y=self.y_spin.value() / 1e3,
                                          z=self.z_spin.value() / 1e3))

    def _on_above_spin(self, v):
        if not self._syncing:
            self.set_trap(self.trap.moved(y=self.gap - v / 1e3))

    def _on_gap(self, v):
        if self._syncing:
            return
        self.gap = v / 1e3
        self.set_trap(self.trap)

    def _on_slider(self, v):
        if not self._syncing:
            self.set_trap(self.trap.moved(y=v / 1e4))

    def _pick_top(self, x_mm, z_mm, phase):
        self.set_trap(self.trap.moved(x=x_mm / 1e3, z=z_mm / 1e3), dragging=(phase == "drag"))
        if phase != "drag":
            self._request_field("fine")

    def _pick_side(self, x_mm, y_mm, phase):
        self.set_trap(self.trap.moved(x=x_mm / 1e3, y=y_mm / 1e3), dragging=(phase == "drag"))
        if phase != "drag":
            self._request_field("fine")

    # ------------------------------------------------------------ output
    def _send(self):
        if self.link is None:
            return
        try:
            self.link.submit_codes(output_codes(self.output, self.trap))
        except Exception as e:  # noqa: BLE001 - never leave a half-state: go dark
            self.output = Output.OFF
            self.link.submit(OFF_FRAME)
            self._update_banner()
            self._alert("critical", "Send failed",
                                           f"Could not build/send the frame; sent ALL OFF.\n\n{e}")

    def all_off(self):
        self.output = Output.OFF
        self._send()
        self._update_banner()
        self._field_changed()

    def all_on(self):
        if not self._require_link():
            return
        self.output = Output.ON
        self._send()
        self._update_banner()
        self._field_changed()

    def emit_trap(self):
        if not self._require_link():
            return
        self.output = Output.TRAP
        self._send()
        self._update_banner()
        self._field_changed()

    def preset_centre(self):
        n = self.preset_spin.value()
        self.set_trap(self.trap.moved(x=0.0, z=0.0, y=self.gap - n / 1e3))
        self._request_field("fine")

    def snap_to_node(self):
        """Move Y to the nearest pressure node of the pattern the trap would emit."""
        model = self._model()
        if not model.reflector:
            self.nodes_label.setText("Snap needs the reflector model on.")
            return
        t = self.trap
        for _ in range(3):   # the pattern shifts slightly with Y; iterate to a fixed point
            src = fld.make_sources(fld.model_phases(output_codes(Output.TRAP, t)), model)
            nodes = fld.pressure_nodes(src, self.gap, t.x, t.z)
            if len(nodes) == 0:
                return
            t = t.moved(y=float(nodes[np.argmin(np.abs(nodes - t.y))]))
        self.set_trap(t)
        self._request_field("fine")

    def _require_link(self):
        if self.link is None:
            self._alert("information", "Not connected",
                                              "Connect first. Nothing was sent.")
            return False
        return True

    # ------------------------------------------------------------ connection
    def toggle_connection(self):
        if self.link is None:
            self.connect_link()
        else:
            self.disconnect_link()

    def connect_link(self):
        port = self.port_combo.currentText().strip()
        try:
            link = self.link_factory(port)
        except Exception as e:  # noqa: BLE001
            self.link_err = f"could not open {port}: {e}"
            self._update_banner()
            self._alert("warning", "Connect failed", self.link_err)
            return
        link.on_error = self.link_error.emit       # sender thread -> queued Qt signal
        self.link, self.link_port, self.link_err = link, port, None
        self.output = Output.OFF
        self.link.submit(OFF_FRAME)                # board now matches the app: dark
        self.connect_btn.setText("Disconnect")
        self.settings.port = port
        self._schedule_save()
        self._update_banner()
        self._field_changed()

    def disconnect_link(self):
        if self.link is not None:
            self.link.close()                      # sends ALL OFF once
        self.link = None
        self.output = Output.OFF
        self.connect_btn.setText("Connect")
        self._update_banner()
        self._field_changed()

    def _on_link_error(self, msg):
        if self.link is not None:
            self.link.close()                      # already tried ALL OFF; just tidy up
        self.link = None
        self.link_err = msg
        self.output = Output.OFF
        self.connect_btn.setText("Connect")
        self._update_banner()
        self._alert("critical", "Link error",
            f"Serial link failed:\n{msg}\n\nALL OFF was attempted once. Board state unknown: "
            "check the PSU current.")

    # ------------------------------------------------------------ status
    def _update_banner(self):
        if self.link is not None:
            mock = self.link_port == MOCK_PORT
            conn = f"MOCK LINK (no hardware)" if mock else f"CONNECTED  {self.link_port}"
            out = {Output.OFF: "OUTPUT: ALL OFF", Output.ON: "OUTPUT: ALL ON (standing wave)",
                   Output.TRAP: f"OUTPUT: TRAP  {self.trap.describe()}"}[self.output]
            bg = "#6a1b9a" if mock else "#1b5e20"
            fg = "white"
            text = f"{conn}   |   {out}"
            light = "#7c4dff" if mock else "#00e676"
            title = f"HAL Control - {'MOCK' if mock else 'CONNECTED'} - {out}"
        elif self.link_err:
            bg, fg, light = "#b71c1c", "white", "#ff1744"
            text = f"LINK ERROR - board state UNKNOWN - {self.link_err}"
            title = "HAL Control - LINK ERROR"
        else:
            bg, fg, light = "#3e2723", "#ffca28", "#757575"
            text = ("NOT CONNECTED - the app is NOT controlling the board "
                    "(its power-on default is ALL ON, so a levitating bead proves nothing)")
            title = "HAL Control - NOT CONNECTED"
        self.banner.setText(text)
        self.banner.setStyleSheet(f"background:{bg};color:{fg};border-radius:6px;padding:6px;")
        self.light.setStyleSheet(f"background:{light};border-radius:11px;border:1px solid #222;")
        self.setWindowTitle(title)
        connected = self.link is not None
        self.on_btn.setEnabled(connected)
        self.emit_btn.setEnabled(connected)
        self.emit_btn.setText("EMITTING TRAP" if self.output is Output.TRAP else "EMIT TRAP")
        self.port_combo.setEnabled(not connected)
        self._update_status()

    def _update_status(self):
        if self.link is None:
            self.stats_label.setText("no link")
            return
        st = self.link.stats()
        if st.last_sent_time:
            age = time.time() - st.last_sent_time
            ts = time.strftime("%H:%M:%S", time.localtime(st.last_sent_time))
            ms = int((st.last_sent_time % 1) * 1000)
            last = f"{ts}.{ms:03d} ({age:.1f} s ago)"
        else:
            last = "none yet"
        self.stats_label.setText(
            f"last frame: {last}\n{st.fps:4.0f} frames/s (max {MAX_FPS:.0f})  "
            f"sent {st.frames_sent}  coalesced {st.frames_coalesced}")

    # ------------------------------------------------------------ field view
    def _model(self):
        return fld.FieldModel(gap=self.gap, reflector=self.refl_cb.isChecked(),
                              refl_coeff=self.refl_spin.value(), vpp=self.vpp_spin.value())

    def _shown_output(self):
        """What the field view shows: what's emitted, or the trap preview when OFF."""
        return Output.TRAP if self.output is Output.OFF else self.output

    def _on_field_settings(self, *_):
        self._schedule_save()
        self._request_field("fine")

    def _field_changed(self):
        if self.dragging:
            self._request_field("coarse")
            self._fine_timer.start()
        else:
            self._request_field("fine")

    def _request_field(self, quality):
        if self.dragging and quality == "fine":
            self._fine_timer.start()
            return
        show = self.show_field_cb.isChecked()
        self.top_img.setVisible(show)
        self.side_img.setVisible(show)
        shown = self._shown_output()
        self.field_label.setText(
            ("Showing: PREVIEW of the trap (not emitting)" if self.output is Output.OFF
             else "Showing: the emitted pattern") + " - quantized phases, model only.")
        self._field_seq += 1
        self.worker.request(FieldRequest(
            seq=self._field_seq, codes=output_codes(shown, self.trap), model=self._model(),
            trap=self.trap, view=self.view_combo.currentData(), quality=quality))

    def _on_field(self, res):
        if res.req.seq < getattr(self, "_shown_seq", 0):
            return
        self._shown_seq = res.req.seq
        self._last_result = res
        view = res.req.view
        if view == "amplitude":
            side, top = res.side.values / 1e3, res.top.values / 1e3     # kPa
            lo, hi = 0.0, float(max(side.max(), top.max(), 1e-9))
            label = "|p| (kPa)"
        else:
            side, top = res.side.values * 1e9, res.top.values * 1e9     # nJ
            both = np.concatenate([side.ravel(), top.ravel()])
            lo, hi = (float(v) for v in np.percentile(both, [1, 99]))
            if hi <= lo:
                hi = lo + 1e-12
            label = "U (nJ)"
        self.cbar.setColorMap(self.cmaps[view])
        self.side_img.setImage(side, autoLevels=False)
        self.top_img.setImage(top, autoLevels=False)
        place_image(self.side_img, res.side)
        place_image(self.top_img, res.top)
        self.cbar.setLevels((lo, hi))
        self.cbar.getAxis("right").setLabel(label)
        self._display = {"side": side, "top": top, "levels": (lo, hi), "label": label,
                         "view": view, "res": res}

        # pressure nodes on the trap's vertical line
        nodes = res.nodes if res.req.model.reflector else np.array([])
        self._nodes = nodes
        tx = res.req.trap.x * 1e3
        self.node_marks.setData([tx] * len(nodes), list(nodes * 1e3))
        if len(nodes):
            above = mm_above_plate(nodes, self.gap)
            near = nodes[np.argmin(np.abs(nodes - self.trap.y))]
            closest = sorted(above[above > 0])[:4]
            self.nodes_label.setText(
                f"Pressure nodes at this X/Z, mm above plate: "
                f"{', '.join(f'{v:.1f}' for v in closest)} ...\n"
                f"Nearest to trap: Y = {near * 1e3:.1f} mm "
                f"({mm_above_plate(near, self.gap):.1f} above plate, "
                f"{(near - self.trap.y) * 1e3:+.1f} mm from trap)")
        else:
            self.nodes_label.setText("Pressure nodes: (reflector model off)")

    # ------------------------------------------------------------ export
    def export_svg(self, which):
        plot = self.side_plot if which == "side" else self.top_plot
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {which} slice", str(Path.home() / f"hal_{which}_slice.svg"),
            "SVG (*.svg)")
        if fn:
            self.write_svg(which, fn)

    def write_svg(self, which, fn):
        Path(fn).write_text(self.slice_figure_svg(which))

    def slice_figure_svg(self, which):
        """SVG of the slice currently on screen (heatmap + overlays + colour bar)."""
        d = getattr(self, "_display", None)
        if d is None:
            raise RuntimeError("no field computed yet")
        res = d["res"]
        t = res.req.trap
        sl = res.side if which == "side" else res.top
        lut = self.cmaps[d["view"]].getLookupTable(nPts=256, alpha=False)
        a, gap_mm = ARRAY_HALF_MM, res.req.model.gap * 1e3
        what = "|p|" if d["view"] == "amplitude" else "Gor'kov U"
        if which == "side":
            lines = [([-a, a], [0, 0], "#4fc3f7", 4),
                     ([-HALF_VIEW_MM, HALF_VIEW_MM], [gap_mm, gap_mm], "#9e9e9e", 4)]
            nodes = res.nodes if res.req.model.reflector else []
            markers = [(t.x * 1e3, n * 1e3, "tri", "#00e5ff") for n in nodes]
            markers.append((t.x * 1e3, t.y * 1e3, "cross", "#00e676"))
            fig = export_mod.SliceFigure(
                d["side"], sl.h * 1e3, sl.v * 1e3, lut, d["levels"],
                title=f"{what}, side view XY at Z = {t.z * 1e3:.1f} mm "
                      f"(gap {gap_mm:.0f} mm) - {t.describe()}",
                xlabel="X (mm)", ylabel="Y (mm from array face)", cbar_label=d["label"],
                v_down=True, view_h=(-HALF_VIEW_MM, HALF_VIEW_MM), view_v=(-4, gap_mm + 4),
                lines=lines, markers=markers)
        else:
            fig = export_mod.SliceFigure(
                d["top"], sl.h * 1e3, sl.v * 1e3, lut, d["levels"],
                title=f"{what}, top view XZ at Y = {t.y * 1e3:.1f} mm "
                      f"({mm_above_plate(t.y, res.req.model.gap):.1f} mm above plate)",
                xlabel="X (mm)", ylabel="Z (mm)", cbar_label=d["label"],
                lines=[([-a, a, a, -a, -a], [-a, -a, a, a, -a], "#4fc3f7", 1.5)],
                markers=[(t.x * 1e3, t.z * 1e3, "cross", "#00e676")])
        return export_mod.slice_svg(fig)

    def export_csv(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export phases", str(Path.home() / "hal_phases.csv"), "CSV (*.csv)")
        if fn:
            self.write_csv(fn)

    def write_csv(self, fn):
        shown = self._shown_output()
        desc = {Output.ON: "all on", Output.OFF: "all off"}.get(shown, self.trap.describe())
        if self.output is Output.OFF:
            desc += " (preview, not emitted)"
        Path(fn).write_text(phases_csv(output_codes(shown, self.trap), desc))

    def _alert(self, level, title, text):
        """Modal message box (tests replace this to avoid blocking)."""
        getattr(QtWidgets.QMessageBox, level)(self, title, text)

    # ------------------------------------------------------------ persistence
    def _schedule_save(self):
        self._save_timer.start()

    def save_settings(self):
        s = self.settings
        s.set_trap(self.trap)
        s.gap_mm = self.gap * 1e3
        s.port = self.port_combo.currentText().strip() or s.port
        if s.port == MOCK_PORT:
            s.port = cfg.Settings.port if not self.settings.port or self.settings.port == MOCK_PORT \
                else self.settings.port
        s.preset_above_plate_mm = self.preset_spin.value()
        s.voltage_note = self.vnote.text()
        s.model_vpp = self.vpp_spin.value()
        s.reflector = self.refl_cb.isChecked()
        s.refl_coeff = self.refl_spin.value()
        s.field_view = self.view_combo.currentData()
        s.show_field = self.show_field_cb.isChecked()
        try:
            cfg.save(s, self.settings_path)
        except OSError as e:
            print(f"could not save settings: {e}", file=sys.stderr)

    def shutdown(self):
        """Idempotent: ALL OFF + close link, stop worker, save."""
        if getattr(self, "_shut", False):
            return
        self._shut = True
        QtWidgets.QApplication.instance().removeEventFilter(self._esc)
        self.disconnect_link()
        self.worker.stop()
        self.save_settings()

    def closeEvent(self, ev):
        self.shutdown()
        super().closeEvent(ev)


class _EscFilter(QtCore.QObject):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.KeyPress and ev.key() == Qt.Key.Key_Escape:
            if not ev.isAutoRepeat():
                self.callback()
            return True
        return False


# =============================================================================
# Entry point
# =============================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m hal_control")
    ap.add_argument("--mock", action="store_true", help="offer a mock link (no hardware)")
    ap.add_argument("--config", default=str(cfg.DEFAULT_PATH))
    args = ap.parse_args(argv)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("HAL Control")
    win = MainWindow(cfg.load(args.config), Path(args.config), mock=args.mock)

    def excepthook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        QtWidgets.QMessageBox.critical(win, "Unexpected error",
                                       "".join(traceback.format_exception_only(etype, value)))
    sys.excepthook = excepthook

    # Ctrl+C in the terminal: close cleanly (ALL OFF). The timer lets Python see signals.
    signal.signal(signal.SIGINT, lambda *_: win.close())
    signal.signal(signal.SIGTERM, lambda *_: win.close())
    tick = QTimer(interval=200)
    tick.timeout.connect(lambda: None)
    tick.start()
    app.aboutToQuit.connect(win.shutdown)

    win.show()
    return app.exec()
