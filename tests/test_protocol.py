import numpy as np
import pytest

from hal_control import protocol as pr


def test_frame_layout_and_channel_order():
    codes = np.arange(256) % 33
    f = pr.build_frame(codes)
    assert len(f) == pr.FRAME_LEN == 259
    assert f[0] == 192 and f[1] == 254 and f[-1] == 253
    for i in range(256):
        assert f[2 + i] == codes[i]  # byte i of the data -> FPGA channel i
    np.testing.assert_array_equal(pr.parse_frame(f), codes)


def test_off_and_on_frames():
    assert pr.parse_frame(pr.OFF_FRAME).tolist() == [32] * 256
    assert pr.parse_frame(pr.ON_FRAME).tolist() == [0] * 256


@pytest.mark.parametrize("bad", [
    np.full(256, 33), np.full(256, -1), np.zeros(255, int), np.zeros(257, int), np.full(256, 1.5),
])
def test_build_frame_rejects_bad_codes(bad):
    with pytest.raises(ValueError):
        pr.build_frame(bad)


def test_float_integer_codes_accepted():
    assert pr.build_frame(np.full(256, 5.0)) == pr.build_frame(np.full(256, 5))


def test_data_never_collides_with_commands():
    f = pr.build_frame(np.full(256, 32))
    assert not ({192, 253, 254} & set(f[2:-1]))


def test_parse_rejects_malformed():
    good = pr.OFF_FRAME
    for bad in (good[:-1], b"\x00" + good[1:], good[:-1] + b"\x00"):
        with pytest.raises(ValueError):
            pr.parse_frame(bad)


def test_frame_time():
    assert pr.FRAME_TIME == pytest.approx(259 * 10 / 230400)
    assert 85 < pr.MAX_FPS < 90
