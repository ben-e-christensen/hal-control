"""File exports that don't need the GUI: phases CSV and slice SVG.

The SVG is written by hand (not pyqtgraph's exporter) so it opens cleanly in
Inkscape: the heatmap is one embedded PNG, everything else (axes, ticks,
labels, array face, plate, nodes, trap marker, colour bar) is real SVG.
"""

import base64
import struct
import zlib
from dataclasses import dataclass, field
from html import escape

import numpy as np

from .geometry import EMITTER_POS, GRID_COL, GRID_ROW
from .phases import OFF_CODE, TWO_PI

CSV_HEADER = "channel,grid_row,grid_col,x_mm,z_mm,code_0_31,phase_rad"


def phases_csv(codes, description=""):
    """CSV text: one row per FPGA channel. code_0_31 is the byte sent (32 = off,
    phase_rad empty). grid_row/col count from the -Z / -X edge."""
    codes = np.asarray(codes)
    lines = []
    if description:
        lines.append(f"# {description}")
    lines.append(CSV_HEADER)
    for ch in range(len(codes)):
        c = int(codes[ch])
        rad = "" if c == OFF_CODE else f"{c * TWO_PI / 32:.6f}"
        lines.append(f"{ch},{GRID_ROW[ch]},{GRID_COL[ch]},{EMITTER_POS[ch, 0] * 1e3:.3f},"
                     f"{EMITTER_POS[ch, 2] * 1e3:.3f},{c},{rad}")
    return "\n".join(lines) + "\n"


# ---- minimal PNG encoder (RGBA8) --------------------------------------------------
def png_rgba(rgba):
    """(H, W, 4) uint8 -> PNG bytes. Row 0 is the top of the image."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def apply_lut(values, lo, hi, lut):
    """Real array -> RGBA via a (N, 3|4) uint8 lookup table."""
    lut = np.asarray(lut, dtype=np.uint8)
    if lut.shape[1] == 3:
        lut = np.hstack([lut, np.full((len(lut), 1), 255, np.uint8)])
    t = (np.asarray(values, float) - lo) / (hi - lo if hi > lo else 1.0)
    idx = np.clip((t * (len(lut) - 1)).round().astype(int), 0, len(lut) - 1)
    return lut[idx]


# ---- slice SVG --------------------------------------------------------------------
@dataclass
class SliceFigure:
    values: np.ndarray            # (len(v), len(h)) display units (kPa, nJ, ...)
    h_mm: np.ndarray              # horizontal sample coordinates
    v_mm: np.ndarray              # vertical sample coordinates
    lut: np.ndarray               # colour map (N, 3|4) uint8
    levels: tuple                 # (lo, hi)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    cbar_label: str = ""
    v_down: bool = False          # True: v increases downward (side view: array on top)
    view_h: tuple = None          # axis ranges in mm (default: data extent)
    view_v: tuple = None
    lines: list = field(default_factory=list)    # (xs, ys, colour, width)
    markers: list = field(default_factory=list)  # (x, y, kind, colour) kind: cross|tri


def _ticks(lo, hi, target=6):
    span = hi - lo
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=mag * 10)
    start = np.ceil(lo / step) * step
    return [round(v, 9) + 0.0 for v in np.arange(start, hi + step * 1e-9, step)]  # no "-0"


def slice_svg(fig, width_px=640):
    """Render a SliceFigure as an SVG string (units: px; data in mm)."""
    h0, h1 = fig.view_h or (fig.h_mm[0], fig.h_mm[-1])
    v0, v1 = fig.view_v or (fig.v_mm[0], fig.v_mm[-1])
    ml, mr, mt, mb, cbw = 64, 96, 34, 50, 14
    pw = width_px - ml - mr
    ph = pw * (v1 - v0) / (h1 - h0)                       # equal aspect: 1 mm = 1 mm
    H = mt + ph + mb

    def X(x):
        return ml + (x - h0) / (h1 - h0) * pw

    def Y(v):
        f = (v - v0) / (v1 - v0)
        return mt + (f if fig.v_down else 1 - f) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width_px}" height="{H:.1f}" '
           f'viewBox="0 0 {width_px} {H:.1f}" font-family="sans-serif" font-size="11">',
           f'<rect x="0" y="0" width="{width_px}" height="{H:.1f}" fill="white"/>']

    # heatmap: pixel centres on the samples
    lo, hi = fig.levels
    rgba = apply_lut(fig.values, lo, hi, fig.lut)
    if not fig.v_down:
        rgba = rgba[::-1]                                  # PNG row 0 = top = max v
    dh = (fig.h_mm[-1] - fig.h_mm[0]) / max(1, len(fig.h_mm) - 1)
    dv = (fig.v_mm[-1] - fig.v_mm[0]) / max(1, len(fig.v_mm) - 1)
    ex0, ex1 = fig.h_mm[0] - dh / 2, fig.h_mm[-1] + dh / 2
    ev0, ev1 = fig.v_mm[0] - dv / 2, fig.v_mm[-1] + dv / 2
    ix, iw = X(ex0), X(ex1) - X(ex0)
    iy = min(Y(ev0), Y(ev1))
    ih = abs(Y(ev1) - Y(ev0))
    b64 = base64.b64encode(png_rgba(rgba)).decode()
    out.append(f'<defs><clipPath id="plot"><rect x="{ml}" y="{mt}" width="{pw:.2f}" '
               f'height="{ph:.2f}"/></clipPath></defs>')
    out.append(f'<g id="heatmap" clip-path="url(#plot)"><image x="{ix:.2f}" y="{iy:.2f}" '
               f'width="{iw:.2f}" height="{ih:.2f}" preserveAspectRatio="none" '
               f'style="image-rendering:optimizeQuality" '
               f'xlink:href="data:image/png;base64,{b64}" href="data:image/png;base64,{b64}"/></g>')

    # overlays
    out.append('<g id="overlays" clip-path="url(#plot)">')
    for xs, ys, colour, w in fig.lines:
        pts = " ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{w}"/>')
    for x, y, kind, colour in fig.markers:
        cx, cy = X(x), Y(y)
        if kind == "cross":
            out.append(f'<g stroke="{colour}" stroke-width="2" fill="none">'
                       f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="6"/>'
                       f'<line x1="{cx - 10:.2f}" y1="{cy:.2f}" x2="{cx + 10:.2f}" y2="{cy:.2f}"/>'
                       f'<line x1="{cx:.2f}" y1="{cy - 10:.2f}" x2="{cx:.2f}" y2="{cy + 10:.2f}"/></g>')
        else:
            out.append(f'<path d="M{cx - 8:.2f},{cy - 4:.2f} L{cx - 8:.2f},{cy + 4:.2f} '
                       f'L{cx - 1:.2f},{cy:.2f} Z" fill="{colour}" stroke="black" '
                       f'stroke-width="0.6"/>')
    out.append('</g>')

    # axes frame, ticks, labels
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw:.2f}" height="{ph:.2f}" fill="none" '
               f'stroke="black" stroke-width="1"/>')
    out.append('<g id="axes" stroke="black" stroke-width="1">')
    for t in _ticks(h0, h1):
        x = X(t)
        out.append(f'<line x1="{x:.2f}" y1="{mt + ph:.2f}" x2="{x:.2f}" y2="{mt + ph + 5:.2f}"/>')
    for t in _ticks(v0, v1):
        y = Y(t)
        out.append(f'<line x1="{ml - 5}" y1="{y:.2f}" x2="{ml}" y2="{y:.2f}"/>')
    out.append('</g><g id="labels" fill="black" stroke="none">')
    for t in _ticks(h0, h1):
        out.append(f'<text x="{X(t):.2f}" y="{mt + ph + 18:.2f}" text-anchor="middle">{t:g}</text>')
    for t in _ticks(v0, v1):
        out.append(f'<text x="{ml - 8}" y="{Y(t) + 4:.2f}" text-anchor="end">{t:g}</text>')
    out.append(f'<text x="{ml + pw / 2:.2f}" y="{H - 10:.2f}" text-anchor="middle" '
               f'font-size="12">{escape(fig.xlabel)}</text>')
    out.append(f'<text transform="translate(16,{mt + ph / 2:.2f}) rotate(-90)" '
               f'text-anchor="middle" font-size="12">{escape(fig.ylabel)}</text>')
    out.append(f'<text x="{ml + pw / 2:.2f}" y="{mt - 12}" text-anchor="middle" '
               f'font-size="13">{escape(fig.title)}</text>')
    out.append('</g>')

    # colour bar
    cx = ml + pw + 18
    bar = apply_lut(np.linspace(hi, lo, 256)[:, None], lo, hi, fig.lut)
    b64 = base64.b64encode(png_rgba(bar)).decode()
    out.append(f'<g id="colorbar"><image x="{cx}" y="{mt}" width="{cbw}" height="{ph:.2f}" '
               f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}" '
               f'href="data:image/png;base64,{b64}"/>'
               f'<rect x="{cx}" y="{mt}" width="{cbw}" height="{ph:.2f}" fill="none" '
               f'stroke="black"/>')
    for t in _ticks(lo, hi, 5):
        y = mt + (hi - t) / (hi - lo if hi > lo else 1) * ph
        out.append(f'<line x1="{cx + cbw}" y1="{y:.2f}" x2="{cx + cbw + 4}" y2="{y:.2f}" '
                   f'stroke="black"/><text x="{cx + cbw + 7}" y="{y + 4:.2f}">{t:g}</text>')
    out.append(f'<text transform="translate({width_px - 8},{mt + ph / 2:.2f}) rotate(-90)" '
               f'text-anchor="middle" font-size="12">{escape(fig.cbar_label)}</text></g>')
    out.append('</svg>')
    return "\n".join(out) + "\n"
