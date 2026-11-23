# HAL Control

PySide6 GUI for the SonicSurface 16×16, 40 kHz phased array (one board facing down, reflector plate below). It replaces Ultraino for day-to-day use. See `CLAUDE.md` for the hardware facts and conventions.

## Run

```
cd ~/Desktop/hal-control
.venv/bin/pytest -q                        # expect 143 passed
.venv/bin/python -m hal_control --mock     # no hardware: pick "mock" in the port box, then Connect
.venv/bin/python -m hal_control            # real board (/dev/ttyUSB0)
```

CLI for quick hardware checks (coordinates in mm):

```
.venv/bin/python -m hal_control.cli --port /dev/ttyUSB0 off
.venv/bin/python -m hal_control.cli --port /dev/ttyUSB0 focus 0 80 0     # Ctrl+C = all off
```

## How the output works

- **Connect** sends ALL OFF straight away.
- **While the output is OFF, trap edits are only a preview.** Nothing is sent until you press **EMIT TRAP**. After that, every change goes out live, throttled to about 85 frames/s.
- **Esc** means ALL OFF from anywhere in the window, even while typing in a spinbox.
- **Disconnect, closing the window, Ctrl+C and link errors** all send ALL OFF.
- On a link error the banner shows **LINK ERROR, board state UNKNOWN**. Check the board yourself.
- Trap type **none** sends ALL OFF.
- The field view is a linear model, computed from the 5-bit phases actually sent. Treat it as a guide, not ground truth.

## Hardware checklist (M2 + M3 GUI, not yet verified)

Watch the PSU current and the Nano RX LED.

- [ ] **1. Launch:** the banner says NOT CONNECTED. Nothing is sent, and the board stays in its power-on state (all on, current high).
- [ ] **2. Connect** (`/dev/ttyUSB0`): the banner turns green with "OUTPUT: ALL OFF". The current drops and the RX LED blinks once.
- [ ] **3. ALL ON:** the current comes back up.
- [ ] **4. Esc:** the current drops again (ALL OFF).
- [ ] **5. EMIT TRAP, then drag on the top view:**
  - the RX LED flickers continuously;
  - frames/s reads about 85 (never above about 89);
  - when you release, frames stop.
- [ ] **6. Disconnect:** the current drops, and the banner says NOT CONNECTED.
- [ ] **7. Close the window:** reconnect, press ALL ON, then close the window. The current should drop.
- [ ] **8. Export:** saving the side/top SVG and the phases CSV works, and the SVG opens in Inkscape.

### Finding a trap (bead test)

- **Bead size:** smaller than about 4 mm (half a wavelength). 1–3 mm styrofoam is easiest.
- **Where it sits:** with one board plus the reflector, the bead sits at the standing-wave pressure nodes, about 4.3 mm apart.
- **Start:** press ALL ON, then place the bead a few mm above the plate near the centre.
- **Then:**
  1. Use the "centre, N mm above plate" preset, or set X/Z to where the bead is.
  2. Click **Snap Y to nearest pressure node**.
  3. Press **EMIT TRAP** and move the trap slowly.
- A trapped bead does **not** prove the app is in control: the power-on default also levitates. Trust the banner and the PSU current.

Notes from checking:

-
