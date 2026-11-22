"""What the app is asking the board to do: output mode + trap description."""

import enum
import math
from dataclasses import dataclass, replace

import numpy as np

from . import phases as ph

KINDS = ("focus", "twin", "vortex", "none")


class Output(enum.Enum):
    OFF = "off"        # all transducers off (code 32)
    ON = "on"          # all on, phase 0: plain standing wave
    TRAP = "trap"      # the trap below (kind "none" = all off)


@dataclass(frozen=True)
class Trap:
    kind: str = "focus"
    x: float = 0.0          # m, app frame
    y: float = 0.08         # m, out of the array face
    z: float = 0.0          # m
    angle_deg: float = 0.0  # twin signature rotation
    charge: int = 1         # vortex topological charge

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown trap kind {self.kind!r}")

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    def moved(self, **kw):
        return replace(self, **kw)

    def phases(self):
        if self.kind == "focus":
            return ph.focus(self.pos)
        if self.kind == "twin":
            return ph.twin(self.pos, math.radians(self.angle_deg))
        if self.kind == "vortex":
            return ph.vortex(self.pos, self.charge)
        return ph.all_off()

    def describe(self):
        s = f"{self.kind} @ ({self.x * 1e3:.1f}, {self.y * 1e3:.1f}, {self.z * 1e3:.1f}) mm"
        if self.kind == "twin":
            s += f", {self.angle_deg:g} deg"
        elif self.kind == "vortex":
            s += f", charge {self.charge:+d}"
        elif self.kind == "none":
            s = "none (all off)"
        return s


def output_phases(output, trap):
    """Radians (NaN = off) for an output mode."""
    if output is Output.OFF:
        return ph.all_off()
    if output is Output.ON:
        return ph.all_on()
    return trap.phases()


def output_codes(output, trap):
    return ph.discretize(output_phases(output, trap))


def mm_above_plate(y, gap):
    """Height of app-frame Y above the reflector plate, in mm."""
    return (gap - y) * 1000.0


def y_for_mm_above_plate(mm, gap):
    return gap - mm / 1000.0
