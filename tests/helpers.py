import numpy as np


def wrap_diff(a, b):
    """Smallest signed angle a - b, elementwise, in (-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))
