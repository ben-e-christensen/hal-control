"""Persistent settings in ~/.config/hal_control.json (human-readable, mm)."""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .trap import KINDS, Trap

DEFAULT_PATH = Path.home() / ".config" / "hal_control.json"


@dataclass
class Settings:
    port: str = "/dev/ttyUSB0"
    gap_mm: float = 120.0
    trap_kind: str = "focus"
    trap_x_mm: float = 0.0
    trap_y_mm: float = 80.0
    trap_z_mm: float = 0.0
    trap_angle_deg: float = 0.0
    trap_charge: int = 1
    preset_above_plate_mm: float = 20.0
    voltage_note: str = ""            # free text, e.g. "12 Vpp bench PSU"
    model_vpp: float = 12.0           # only scales the field view
    reflector: bool = True
    refl_coeff: float = 1.0
    field_view: str = "amplitude"     # or "gorkov"
    show_field: bool = True

    def trap(self):
        return Trap(kind=self.trap_kind, x=self.trap_x_mm / 1e3, y=self.trap_y_mm / 1e3,
                    z=self.trap_z_mm / 1e3, angle_deg=self.trap_angle_deg,
                    charge=self.trap_charge)

    def set_trap(self, t):
        self.trap_kind = t.kind
        self.trap_x_mm, self.trap_y_mm, self.trap_z_mm = t.x * 1e3, t.y * 1e3, t.z * 1e3
        self.trap_angle_deg = t.angle_deg
        self.trap_charge = t.charge


def _coerce(default, value):
    """Value converted to default's type, or raise."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise TypeError
    if isinstance(default, int):
        if isinstance(value, bool) or not float(value).is_integer():
            raise TypeError
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool):
            raise TypeError
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError
        return v
    if isinstance(default, str):
        if not isinstance(value, str):
            raise TypeError
        return value
    raise TypeError


def load(path=DEFAULT_PATH):
    """Settings from disk. Missing file, bad JSON or bad keys fall back to defaults."""
    s = Settings()
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return s
    if not isinstance(data, dict):
        return s
    for f in fields(Settings):
        if f.name in data:
            try:
                setattr(s, f.name, _coerce(getattr(s, f.name), data[f.name]))
            except (TypeError, ValueError):
                pass
    if s.trap_kind not in KINDS:
        s.trap_kind = Settings.trap_kind
    if s.field_view not in ("amplitude", "gorkov"):
        s.field_view = Settings.field_view
    return s


def save(settings, path=DEFAULT_PATH):
    """Atomic write (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2) + "\n")
    os.replace(tmp, path)
