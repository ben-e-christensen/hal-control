"""SonicSurface serial frame building. Pure functions, no I/O.

Frame: 192 (enable primary board), 254 (start phases), 256 codes in FPGA
channel order, 253 (commit / swap buffers). Codes 0..31 = phase steps of
2*pi/32, 32 = off.
"""

import numpy as np

from .geometry import N_EMITTERS
from .phases import OFF_CODE, all_off, all_on, discretize

BAUD = 230400
BITS_PER_BYTE = 10  # 8N1

CMD_ENABLE_PRIMARY = 192
CMD_START_PHASES = 254
CMD_COMMIT = 253

HEADER = bytes([CMD_ENABLE_PRIMARY, CMD_START_PHASES])
TRAILER = bytes([CMD_COMMIT])
FRAME_LEN = len(HEADER) + N_EMITTERS + len(TRAILER)  # 259
FRAME_TIME = FRAME_LEN * BITS_PER_BYTE / BAUD          # ~11.24 ms on the wire
MAX_FPS = 1.0 / FRAME_TIME                             # ~89 frames/s theoretical


def build_frame(codes):
    """256 codes (0..32, FPGA order) -> frame bytes."""
    c = np.asarray(codes)
    if c.shape != (N_EMITTERS,):
        raise ValueError(f"need {N_EMITTERS} codes, got shape {c.shape}")
    if not np.issubdtype(c.dtype, np.integer):
        if not np.all(np.mod(c, 1) == 0):
            raise ValueError("codes must be integers")
    if c.min() < 0 or c.max() > OFF_CODE:
        raise ValueError(f"codes must be in 0..{OFF_CODE}, got {c.min()}..{c.max()}")
    return HEADER + bytes(c.astype(np.uint8)) + TRAILER


def frame_from_phases(phases):
    """Radians (NaN = off) -> frame bytes."""
    return build_frame(discretize(phases))


def parse_frame(frame):
    """Frame bytes -> codes array. Raises ValueError if malformed."""
    frame = bytes(frame)
    if len(frame) != FRAME_LEN:
        raise ValueError(f"frame length {len(frame)} != {FRAME_LEN}")
    if not frame.startswith(HEADER) or not frame.endswith(TRAILER):
        raise ValueError("bad header/trailer")
    codes = np.frombuffer(frame[len(HEADER):-len(TRAILER)], dtype=np.uint8).copy()
    if codes.max() > OFF_CODE:
        raise ValueError("code > 32 in frame")
    return codes


OFF_FRAME = frame_from_phases(all_off())
ON_FRAME = frame_from_phases(all_on())
