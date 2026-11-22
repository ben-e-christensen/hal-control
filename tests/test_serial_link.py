import time

import numpy as np
import pytest

from hal_control.protocol import FRAME_TIME, OFF_FRAME, ON_FRAME, build_frame
from hal_control.serial_link import MockLink, SerialLink


def numbered_frame(i):
    c = np.full(256, i % 32)
    c[0] = i // 32
    return build_frame(c)


def test_single_frame_is_sent():
    link = MockLink()
    link.submit(ON_FRAME)
    assert link.wait_idle(1.0)
    assert link.sent == [ON_FRAME]
    st = link.stats()
    assert st.connected and st.frames_sent == 1 and st.last_sent_time > 0
    link.close()


def test_submit_never_blocks_and_coalesces_to_latest():
    link = MockLink(write_time=0.02)
    t0 = time.perf_counter()
    for i in range(200):
        link.submit(numbered_frame(i))
    assert time.perf_counter() - t0 < 0.05    # 200 submits while writes take 20 ms each
    assert link.wait_idle(2.0)
    assert len(link.sent) <= 3                # first, maybe one in flight, latest
    assert link.sent[-1] == numbered_frame(199)
    assert link.stats().frames_coalesced >= 196
    link.close()


def test_rate_never_exceeds_link():
    link = MockLink()
    end = time.monotonic() + 0.4
    i = 0
    while time.monotonic() < end:
        link.submit(numbered_frame(i))
        i += 1
        time.sleep(0.001)
    link.wait_idle(1.0)
    ts = np.array([t for t, _ in link.frames])
    assert len(ts) >= 10
    assert np.diff(ts).min() >= 0.95 * FRAME_TIME
    assert link.stats().fps <= 1.0 / FRAME_TIME + 1
    link.close()


def test_close_sends_off_exactly_once():
    link = MockLink()
    link.submit(ON_FRAME)
    link.wait_idle(1.0)
    link.close()
    link.close()
    assert link.sent == [ON_FRAME, OFF_FRAME]
    assert link.port_closed
    assert not link.connected
    assert link.submit(ON_FRAME) is False


def test_close_drops_pending_frame_and_still_sends_off():
    link = MockLink(write_time=0.05)
    link.submit(numbered_frame(1))
    time.sleep(0.01)                  # first write in flight
    link.submit(numbered_frame(2))    # pending, must not go out after close
    link.close()
    assert link.sent == [numbered_frame(1), OFF_FRAME]


def test_write_failure_sends_off_and_closes():
    errors = []
    link = MockLink(fail_writes={0}, on_error=errors.append)
    link.submit(ON_FRAME)
    assert link.wait_idle(1.0) is False
    deadline = time.monotonic() + 1.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors and "mock write 0 failed" in errors[0]
    assert link.sent == [OFF_FRAME]   # write 0 raised; write 1 was the all-off
    assert link.port_closed and not link.connected
    assert link.stats().error
    link.close()                      # no second all-off on an already-failed link
    assert link.sent == [OFF_FRAME]


def test_serial_link_writes_frames_over_loopback():
    pytest.importorskip("serial")
    link = SerialLink("loop://")
    link.submit(ON_FRAME)
    assert link.wait_idle(1.0)
    assert link._ser.read(len(ON_FRAME)) == ON_FRAME
    link.close()
    assert not link.connected
