# HAL Control — CLAUDE.md

Python desktop GUI to control a SonicSurface 16×16 ultrasonic phased array (40 kHz) for acoustic levitation. It replaces Ultraino (Java) for day-to-day use: connect, on/off, place and drag traps at exact coordinates, see a live field slice that includes the reflector plate.

## Who this is for
- Busta (Ben), PSU, building HAL for tribocharging research. Comfortable with hardware/firmware and Python.
- Keep explanations short and practical. When giving terminal steps, use plain text and code blocks.
- Ask before big architectural changes. Prefer small, working increments over big rewrites.

## Machine
- Ubuntu 24.04 (lab tower). Python 3.12. User is in `dialout`, and `brltty` is removed.
- USB-serial: a DFRduino Nano running an empty sketch, used as a pass-through (CH340). Usually `/dev/ttyUSB0`. Its RX0 pin feeds the board's RX.
- Ultraino stays installed at `~/ultraino/bin2.5` as a fallback. Don't touch it.

## Hardware facts (verified from the firmware and Ultraino source; don't "fix" these without asking)
- **Baud: 230400, 8N1.** The FPGA UART runs at 50 MHz × 128/625 = 10.24 MHz with CLKS_PER_BIT = 45, which is ~227.6 kbaud. Ultraino's chainedFPGA protocol also uses 230400.
- **Frame:** `254` (start phases), then 256 phase bytes in FPGA channel order, then `253` (commit/swap). Ultraino also sends `192` (enable primary board) before the data. Sending it first is harmless and makes the board robust to a reset.
- **Phase byte:** 0–31 = phase in steps of 2π/32 (11.25°). **32 = off.** Discretize by rounding (Ultraino: `round(phase_rad / (2π) * 32) mod 32`), not by truncation.
- **Channel order:** byte *i* of the frame goes to FPGA channel *i*. In `refs/SonicSurface.py`, `EMITTERS_POS` is already listed in FPGA channel order (its `sendPhases` sends without permuting). Treat it as the geometry source of truth, and cross-check it against Ultraino's `emptyboard.xml.gz`, where transducer `orderNumber` = channel.
- **Calibration lives in the FPGA** (`PHASE_CORRECTION` in `Distribute.vhd`, flashed as `board1_calibrated.jic`). The app sends *uncalibrated* phases. **Never apply calibration in software.** Optional per-board offsets may come later, behind an explicit off-by-default setting.
- **Update rate:** 258 bytes ≈ 11.2 ms per frame, so at most ~85 frames/s. Throttle sends: coalesce to the latest frame and never queue a backlog.
- **Power-on default:** the FPGA boots with every transducer on at phase 0 (a standing wave against the plate). A trapped bead does **not** prove the app is in control. The status UI must make the real connection state obvious.
- Wavelength 8.65 mm (air, 25 °C). Element pitch ~10.5 mm. The array is in the plane Y = 0.

## Coordinate conventions
- **App/internal frame = SonicSurface.py/Ultraino frame:** origin at the array-face centre, **+Y = straight out of the array face** (toward the reflector), X/Z lateral, SI units (m).
- Physical rig: the board faces **down** and the reflector plate sits at **Y = gap** (user setting, typically 0.04–0.16 m). The UI shows "mm above plate" (= (gap − Y) × 1000) next to raw coordinates.
- **`refs/hal_lev_sim.py` uses a different frame.** In its "single + reflector" setup, z = 0 is the **reflector** and the board sits at z = H facing down, with x/y lateral, in mm in its GUI and m internally. Its point (x_s, y_s, z_s) maps to ours as **X = x_s, Z = y_s, Y = H − z_s**. Its emitters are in plain **raster order** (row-major 16×16 grid), not FPGA channel order. Re-map them through `geometry.py` before sending anything. Convert at the boundary only; don't mix the conventions inside the app.

## Physics model (reuse, don't reinvent)
- Reuse the piston-source model, directivity and Gor'kov potential from `refs/hal_lev_sim.py` where possible, ported into `hal_control/field.py` with the axis conversion.
- **Reflector:** image-source method. Mirror every emitter across the plane Y = gap (same phase, rigid reflector, optional reflection coefficient). Free field = reflector disabled.
- Trap phase recipes: focus (from `phasesForFocus`), twin trap (focus + π on the half-array with x > 0, rotatable signature axis), vortex (focus + charge·atan2). Get the sign convention right against the existing references and lock it with tests.
- The field slice must stay interactive: vectorized numpy, a coarse grid while dragging and a fine grid on release.

## Layout
```
hal_control/
  geometry.py     # emitter positions (FPGA order), constants
  phases.py       # focus / twin / vortex / off, discretize()
  protocol.py     # frame building (pure functions, no I/O)
  serial_link.py  # SerialLink + MockLink (records frames), throttled sender thread
  cli.py          # hardware test CLI (off/on/focus/twin/vortex)
  field.py        # pressure on a slice, image-source reflector, optional Gor'kov
  trap.py         # Output (off/on/trap) + Trap dataclass -> phases/codes
  export.py       # phases CSV, hand-written slice SVG (Inkscape-friendly)
  app.py          # PySide6 GUI (pyqtgraph for slices)
  config.py       # gap, port, voltage note, last trap -> ~/.config/hal_control.json
refs/             # READ-ONLY reference code (SonicSurface.py, hal_lev_sim.py, ultraino notes)
tests/
```

## Stack
- PySide6 + pyqtgraph + numpy + pyserial. No other heavy dependencies without asking.
- `python -m hal_control` launches the app. `--mock` runs without hardware, using MockLink.
- Use a venv (`python3 -m venv .venv`). Don't install packages system-wide.

## Testing rules
- Unit tests (pytest) for everything that isn't GUI: discretize rounding and wrap, 32 = off, frame bytes and length, channel order, axis conversion, focus phases vs `refs/SonicSurface.py`, twin/vortex vs the references, image-source symmetry.
- **Golden test:** the app's focus frame for a given point must equal what `refs/SonicSurface.py` would send for the same point (after the rounding-vs-truncation difference is accounted for).
- Hardware tests are run by the user. Give the exact command and what to look for (PSU current change on all-off, the Nano RX LED blinking, bead behaviour). Never claim hardware works without the user confirming.

## Safety / don'ts
- On close, on disconnect, and on an unhandled exception in the send path: send all-off (32s) once before closing the port.
- Don't send frames faster than the link allows. Don't block the GUI thread on serial I/O.
- Don't write to the firmware, the Quartus project or the Ultraino install.
- 40 kHz is inaudible but very loud at the focus. Don't add features that run at full power unprompted, e.g. auto-sweep on launch.
