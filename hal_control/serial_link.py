"""Serial links with a throttled, coalescing sender thread.

submit(frame) never blocks: it replaces the single pending slot, so if the
caller produces frames faster than the wire allows, intermediate frames are
dropped and only the newest is sent. A background thread sends at most one
frame per FRAME_TIME.

Safety: close() sends OFF_FRAME once before closing the port. If a write
fails, the sender tries OFF_FRAME once, closes the port and enters the
error state (on_error callback is called from the sender thread).
"""

import collections
import threading
import time
from dataclasses import dataclass

from .protocol import BAUD, FRAME_TIME, OFF_FRAME, build_frame, frame_from_phases


@dataclass
class LinkStats:
    connected: bool
    frames_sent: int
    frames_coalesced: int      # submitted but superseded before sending
    last_sent_time: float      # time.time() of the last completed write, 0 if none
    fps: float                 # frames written in the last second
    error: str | None


class Link:
    """Base class. Subclasses implement _write(frame) and _close_port()."""

    def __init__(self, min_interval=FRAME_TIME, on_error=None):
        self.min_interval = min_interval
        self.on_error = on_error
        self._cond = threading.Condition()
        self._pending = None
        self._stopping = False
        self._closed = False
        self._busy = False
        self._error = None
        self._frames_sent = 0
        self._coalesced = 0
        self._last_sent_time = 0.0
        self._recent = collections.deque()
        self._next_allowed = 0.0
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()

    # ---- public API -------------------------------------------------------
    @property
    def connected(self):
        with self._cond:
            return not self._closed and self._error is None

    def submit(self, frame):
        """Queue a frame (bytes). Replaces any frame not yet sent. Non-blocking."""
        frame = bytes(frame)
        with self._cond:
            if self._closed or self._error is not None:
                return False
            if self._pending is not None:
                self._coalesced += 1
            self._pending = frame
            self._cond.notify_all()
            return True

    def submit_codes(self, codes):
        return self.submit(build_frame(codes))

    def submit_phases(self, phases):
        return self.submit(frame_from_phases(phases))

    def wait_idle(self, timeout=1.0):
        """Block until nothing is pending or being written. True if idle."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._pending is not None or self._busy:
                if self._error is not None or self._closed:
                    return False
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
            return self._error is None

    def stats(self):
        with self._cond:
            now = time.monotonic()
            while self._recent and now - self._recent[0] > 1.0:
                self._recent.popleft()
            return LinkStats(
                connected=not self._closed and self._error is None,
                frames_sent=self._frames_sent,
                frames_coalesced=self._coalesced,
                last_sent_time=self._last_sent_time,
                fps=float(len(self._recent)),
                error=self._error,
            )

    def close(self, send_off=True):
        """Stop the sender, send all-off once (unless the link failed), close."""
        with self._cond:
            if self._closed:
                return
            self._stopping = True
            self._pending = None
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        with self._cond:
            failed = self._error is not None
            self._closed = True
            self._cond.notify_all()
        if not failed:
            if send_off:
                try:
                    self._write(OFF_FRAME)
                    self._record_sent()
                except Exception:
                    pass
            self._safe_close_port()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- sender thread ----------------------------------------------------
    def _run(self):
        while True:
            with self._cond:
                while not self._stopping and (
                    self._pending is None or time.monotonic() < self._next_allowed
                ):
                    if self._pending is None:
                        self._cond.wait()
                    else:
                        self._cond.wait(max(0.0, self._next_allowed - time.monotonic()))
                if self._stopping:
                    return
                frame, self._pending = self._pending, None
                self._busy = True
            t0 = time.monotonic()
            try:
                self._write(frame)
            except Exception as e:  # noqa: BLE001 - any failure is fatal for the link
                self._fail(e)
                return
            with self._cond:
                self._busy = False
                self._next_allowed = t0 + self.min_interval
                self._record_sent_locked()
                self._cond.notify_all()

    def _record_sent(self):
        with self._cond:
            self._record_sent_locked()

    def _record_sent_locked(self):
        self._frames_sent += 1
        self._last_sent_time = time.time()
        self._recent.append(time.monotonic())

    def _fail(self, exc):
        try:
            self._write(OFF_FRAME)
        except Exception:
            pass
        self._safe_close_port()
        with self._cond:
            self._error = f"{type(exc).__name__}: {exc}"
            self._busy = False
            self._pending = None
            self._cond.notify_all()
        if self.on_error is not None:
            try:
                self.on_error(self._error)
            except Exception:
                pass

    def _safe_close_port(self):
        try:
            self._close_port()
        except Exception:
            pass

    # ---- subclass hooks ---------------------------------------------------
    def _write(self, frame):
        raise NotImplementedError

    def _close_port(self):
        pass


class SerialLink(Link):
    """Real board via a USB-serial bridge (230400 8N1)."""

    def __init__(self, port, baud=BAUD, write_timeout=1.0, **kw):
        import serial  # pyserial; imported here so MockLink works without it

        self.port = port
        # serial_for_url accepts plain device paths and also test URLs like loop://
        self._ser = serial.serial_for_url(
            port, baudrate=baud, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=0, write_timeout=write_timeout,
        )
        super().__init__(**kw)

    def _write(self, frame):
        n = self._ser.write(frame)
        if n != len(frame):
            raise IOError(f"short write: {n}/{len(frame)} bytes")
        self._ser.flush()  # wait until the OS has transmitted it: no kernel backlog

    def _close_port(self):
        self._ser.close()


class MockLink(Link):
    """Records frames instead of sending them.

    write_time : seconds each write takes (simulates the wire).
    fail_writes: set of 0-based write-call indices that raise IOError.
    """

    def __init__(self, write_time=0.0, fail_writes=(), **kw):
        self.write_time = write_time
        self.fail_writes = set(fail_writes)
        self.frames = []          # list of (time.monotonic(), bytes)
        self.write_calls = 0
        self.port_closed = False
        super().__init__(**kw)

    def _write(self, frame):
        i = self.write_calls
        self.write_calls += 1
        if self.write_time:
            time.sleep(self.write_time)
        if i in self.fail_writes:
            raise IOError(f"mock write {i} failed")
        self.frames.append((time.monotonic(), bytes(frame)))

    def _close_port(self):
        self.port_closed = True

    @property
    def sent(self):
        return [f for _, f in self.frames]
