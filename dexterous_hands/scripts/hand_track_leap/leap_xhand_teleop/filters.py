"""
One-Euro filter (Casiez et al. 2012) for low-latency keypoint smoothing.

The filter cuts jitter at low speeds while staying responsive at high speeds,
which is what we want for teleop -- aggressive smoothing on a still hand,
minimal lag during fast moves.

Reference: https://gery.casiez.net/1euro/
"""
from __future__ import annotations
import math
import numpy as np


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilterScalar:
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        t_e = max(t - self._t_prev, 1e-6)
        a_d = _smoothing_factor(t_e, self.dcutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class OneEuroFilterVec:
    """Vectorized One-Euro filter for arbitrary-shaped numpy arrays."""

    def __init__(self, shape, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.shape = shape
        self._x_prev: np.ndarray | None = None
        self._dx_prev = np.zeros(shape, dtype=np.float32)
        self._t_prev: float | None = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = np.zeros(self.shape, dtype=np.float32)
        self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = x.astype(np.float32)
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x.copy()
            return x
        t_e = max(t - self._t_prev, 1e-6)
        a_d = _smoothing_factor(t_e, self.dcutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        # Per-element cutoff scaled by speed magnitude.
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = (2.0 * math.pi * cutoff * t_e) / (2.0 * math.pi * cutoff * t_e + 1.0)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class KeypointFilter:
    """Convenience wrapper: a One-Euro filter sized for (21, 3) keypoints.

    Tuned defaults for ~30 Hz teleop:
      mincutoff=1.5  -> ~1.5 Hz floor (kills static jitter)
      beta=0.05      -> moderate adaptation to fast motion
    Bump beta up for crisper response, mincutoff up if it feels laggy.
    """

    def __init__(self, mincutoff: float = 1.5, beta: float = 0.05, freq: float = 30.0):
        self._f = OneEuroFilterVec(shape=(21, 3), freq=freq,
                                   mincutoff=mincutoff, beta=beta)

    def reset(self):
        self._f.reset()

    def filter(self, points: np.ndarray, t: float) -> np.ndarray:
        return self._f(points, t)
