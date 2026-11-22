import json
import struct
import xml.dom.minidom
import zlib

import numpy as np
import pytest

from hal_control import config as cfg
from hal_control import export
from hal_control import phases as ph
from hal_control.trap import Output, Trap, mm_above_plate, output_codes, y_for_mm_above_plate


# ---- trap -----------------------------------------------------------------------
def test_output_codes():
    t = Trap("twin", 0.01, 0.06, -0.02, angle_deg=90)
    assert (output_codes(Output.OFF, t) == 32).all()
    assert (output_codes(Output.ON, t) == 0).all()
    np.testing.assert_array_equal(output_codes(Output.TRAP, t),
                                  ph.discretize(ph.twin(t.pos, np.pi / 2)))
    v = Trap("vortex", 0, 0.08, 0, charge=-2)
    np.testing.assert_array_equal(output_codes(Output.TRAP, v),
                                  ph.discretize(ph.vortex(v.pos, -2)))
    assert (output_codes(Output.TRAP, Trap("none")) == 32).all()


def test_trap_validation_and_heights():
    with pytest.raises(ValueError):
        Trap("laser")
    assert mm_above_plate(0.08, 0.12) == pytest.approx(40.0)
    assert y_for_mm_above_plate(40.0, 0.12) == pytest.approx(0.08)
    assert "charge -2" in Trap("vortex", charge=-2).describe()


# ---- config ---------------------------------------------------------------------
def test_config_round_trip(tmp_path):
    p = tmp_path / "sub" / "hal.json"
    s = cfg.Settings(port="/dev/ttyUSB1", gap_mm=95.5, preset_above_plate_mm=12.0)
    s.set_trap(Trap("vortex", 0.012, 0.05, -0.003, charge=2))
    cfg.save(s, p)
    back = cfg.load(p)
    assert back == s
    t = back.trap()
    assert t.kind == "vortex" and t.charge == 2
    np.testing.assert_allclose(t.pos, [0.012, 0.05, -0.003])


def test_config_defaults_on_missing_or_corrupt(tmp_path):
    assert cfg.load(tmp_path / "nope.json") == cfg.Settings()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cfg.load(bad) == cfg.Settings()
    bad.write_text("[1, 2]")
    assert cfg.load(bad) == cfg.Settings()


def test_config_bad_values_fall_back_per_key(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"gap_mm": "wide", "port": "/dev/ttyACM0", "trap_kind": "laser",
                             "trap_charge": 1.5, "reflector": "yes", "refl_coeff": 0.7,
                             "unknown": 3, "field_view": "phase"}))
    s = cfg.load(p)
    d = cfg.Settings()
    assert s.port == "/dev/ttyACM0" and s.refl_coeff == 0.7
    assert (s.gap_mm, s.trap_kind, s.trap_charge, s.reflector, s.field_view) == (
        d.gap_mm, d.trap_kind, d.trap_charge, d.reflector, d.field_view)


# ---- export ---------------------------------------------------------------------
def test_phases_csv():
    codes = ph.discretize(ph.focus(np.array([0, 0.08, 0])))
    codes[5] = 32
    text = export.phases_csv(codes, "focus test")
    lines = text.splitlines()
    assert lines[0] == "# focus test" and lines[1] == export.CSV_HEADER
    rows = [l.split(",") for l in lines[2:]]
    assert len(rows) == 256
    assert [int(r[0]) for r in rows] == list(range(256))
    assert [int(r[5]) for r in rows] == codes.tolist()
    assert rows[5][6] == "" and float(rows[0][6]) == pytest.approx(codes[0] * 2 * np.pi / 32)


def _decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        crc = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        chunks[tag] = body
        pos += 12 + n
    w, h = struct.unpack(">II", chunks[b"IHDR"][:8])
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [raw[i * (4 * w + 1) + 1:(i + 1) * (4 * w + 1)] for i in range(h)]
    return np.frombuffer(b"".join(rows), np.uint8).reshape(h, w, 4)


def test_png_encoder_round_trip():
    img = np.random.default_rng(0).integers(0, 256, (7, 5, 4), dtype=np.uint8)
    np.testing.assert_array_equal(_decode_png(export.png_rgba(img)), img)


def test_slice_svg_is_valid_and_labelled():
    v = np.linspace(0, 1, 12)[:, None] * np.ones((1, 20))
    fig = export.SliceFigure(v, np.linspace(-95, 95, 20), np.linspace(1, 119, 12),
                             np.array([[0, 0, 0], [255, 255, 255]], np.uint8), (0.0, 1.0),
                             title="side & test", xlabel="X (mm)", ylabel="Y (mm)",
                             cbar_label="|p| (kPa)", v_down=True,
                             lines=[([-84, 84], [0, 0], "#4fc3f7", 4)],
                             markers=[(0, 80, "cross", "#00e676"), (0, 60, "tri", "#00e5ff")])
    doc = xml.dom.minidom.parseString(export.slice_svg(fig))
    texts = [t.firstChild.data for t in doc.getElementsByTagName("text") if t.firstChild]
    assert "side & test" in texts and "X (mm)" in texts and "|p| (kPa)" in texts
    assert len(doc.getElementsByTagName("image")) == 2      # heatmap + colour bar
    assert doc.getElementsByTagName("polyline")
