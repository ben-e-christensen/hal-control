"""Minimal hardware test CLI.

  python -m hal_control.cli --port /dev/ttyUSB0 off
  python -m hal_control.cli --port /dev/ttyUSB0 on
  python -m hal_control.cli --port /dev/ttyUSB0 focus X Y Z
  python -m hal_control.cli --port /dev/ttyUSB0 twin X Y Z [--angle DEG]
  python -m hal_control.cli --port /dev/ttyUSB0 vortex X Y Z [--charge N] [--rotation DEG]

X, Y, Z in mm, app frame: origin at the array-face centre, +Y out of the face
toward the plate. 'off' sends all-off and exits. Every other command sends its
pattern and holds until Ctrl+C, then sends all-off (the link always sends
all-off on close). --mock prints the frame instead of opening a port.
"""

import argparse
import math
import signal
import sys
import time

import numpy as np

from . import phases as ph
from .protocol import build_frame, parse_frame
from .serial_link import MockLink, SerialLink


def build_parser():
    p = argparse.ArgumentParser(prog="python -m hal_control.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--mock", action="store_true", help="don't open a port; print the frame")
    p.add_argument("--gap", type=float, default=None,
                   help="array-to-plate gap in mm (only used to print 'mm above plate')")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("off", help="all transducers off (code 32)")
    sub.add_parser("on", help="all transducers on, phase 0")
    for name in ("focus", "twin", "vortex"):
        s = sub.add_parser(name)
        s.add_argument("x", type=float, help="mm")
        s.add_argument("y", type=float, help="mm, distance out of the array face")
        s.add_argument("z", type=float, help="mm")
        if name == "twin":
            s.add_argument("--angle", type=float, default=0.0, help="signature rotation, degrees")
        if name == "vortex":
            s.add_argument("--charge", type=int, default=1)
            s.add_argument("--rotation", type=float, default=0.0, help="degrees")
    return p


def codes_for(args):
    if args.cmd == "off":
        return ph.discretize(ph.all_off())
    if args.cmd == "on":
        return ph.discretize(ph.all_on())
    if args.y <= 0:
        raise SystemExit("Y must be > 0 mm (it is the distance out of the array face).")
    p = np.array([args.x, args.y, args.z]) / 1000.0
    if args.cmd == "focus":
        rad = ph.focus(p)
    elif args.cmd == "twin":
        rad = ph.twin(p, math.radians(args.angle))
    else:
        rad = ph.vortex(p, args.charge, math.radians(args.rotation))
    return ph.discretize(rad)


def describe(args, codes):
    s = f"{args.cmd}"
    if args.cmd not in ("on", "off"):
        s += f" at X={args.x:g} Y={args.y:g} Z={args.z:g} mm"
        if args.gap is not None:
            s += f" ({args.gap - args.y:g} mm above plate)"
        if args.cmd == "twin":
            s += f", angle {args.angle:g} deg"
        if args.cmd == "vortex":
            s += f", charge {args.charge}, rotation {args.rotation:g} deg"
    n_off = int(np.sum(codes == ph.OFF_CODE))
    s += f" | {256 - n_off} on, {n_off} off"
    return s


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None, out=sys.stdout):
    args = build_parser().parse_args(argv)
    codes = codes_for(args)
    frame = build_frame(codes)
    print(describe(args, codes), file=out)

    if args.mock:
        link = MockLink()
    else:
        try:
            link = SerialLink(args.port)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not open {args.port}: {e}", file=out)
            return 2

    try:
        link.submit(frame)
        if not link.wait_idle(timeout=2.0):
            print(f"ERROR: send failed: {link.stats().error}", file=out)
            return 1
        print(f"sent {len(frame)} bytes to {'mock' if args.mock else args.port}", file=out)
        if args.mock:
            grid = parse_frame(link.sent[-1]).reshape(16, 16)
            print("codes, FPGA channel order (row = 16 channels):", file=out)
            for row in grid:
                print(" ".join(f"{c:2d}" for c in row), file=out)
            return 0
        if args.cmd == "off":
            return 0
        print("Holding. Ctrl+C to send ALL OFF and exit.", file=out)
        signal.signal(signal.SIGTERM, _raise_interrupt)  # `kill` also switches off
        try:
            while link.connected:
                time.sleep(0.2)
            print(f"ERROR: link dropped: {link.stats().error}", file=out)
            return 1
        except KeyboardInterrupt:
            print("\nCtrl+C: sending ALL OFF.", file=out)
            return 0
    finally:
        link.close()  # sends all-off once (no-op if the link already failed)


if __name__ == "__main__":
    sys.exit(main())
