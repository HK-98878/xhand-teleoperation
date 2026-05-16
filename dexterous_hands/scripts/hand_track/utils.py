import numpy as np
from typing import Optional, Union

import roslibpy

# --- one-euro filter -------------------------------------------------------

class OneEuroScalar:
    """One-euro filter for a single scalar value.

    Reference: Casiez, Roussel, Vogel (2012), "1€ Filter".

    Parameters:
      min_cutoff: cutoff frequency at zero velocity. Lower = smoother but
                  more lag at rest. Sensible range 0.3-2.0 Hz.
      beta:       how aggressively cutoff increases with speed. Higher =
                  less lag during fast motion but more jitter passes through.
                  Sensible range 0.001-0.1.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        
        assert self.x_prev is not None

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroVector:
    """Per-element one-euro filter for an arbitrary-shape array. Applies
    the same parameters to every element. Shape is fixed on first call."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self._params = (min_cutoff, beta, d_cutoff)
        self._filters: Optional[np.ndarray] = None
        self._shape = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._filters is None:
            self._shape = x.shape
            self._filters = np.empty(x.size, dtype=object)
            for i in range(x.size):
                self._filters[i] = OneEuroScalar(*self._params)
        flat_in = x.reshape(-1)
        flat_out = np.empty_like(flat_in)
        for i, val in enumerate(flat_in):
            flat_out[i] = self._filters[i](float(val), t)
        return flat_out.reshape(self._shape)


# ---------------------------------------------------------------------------
# ROS publisher
# ---------------------------------------------------------------------------
class XHandPublisher:
    JOINT_NAMES = [
        "thumb_bend_joint",
        "thumb_rota_joint1",
        "thumb_rota_joint2",
        "index_bend_joint",
        "index_joint1",
        "index_joint2",
        "mid_joint1",
        "mid_joint2",
        "ring_joint1",
        "ring_joint2",
        "pinky_joint1",
        "pinky_joint2",
    ]
    JOINT_LIMITS = np.array([
        [0, 1.57],
        [-1.05, 1.57],
        [0, 1.57],
        [-0.09, 0.3],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
        [0, 1.92],
    ])

    cmd_numeric_types = Union[int, float, list[int], list[float], np.ndarray]
    @staticmethod
    def __convert_to_command_json_type(value : cmd_numeric_types, target_type : type):
        if isinstance(value, (int, float)):
            return [target_type(value)]*12
        elif (isinstance(value, list) and len(value) == 12) or (isinstance(value, np.ndarray) and value.size == (12,)):
            return [target_type(i) for i in value]
        return []

    def __init__(self, host='localhost', port=9090, hand_id=0,
                 kp : cmd_numeric_types = 100.0, 
                 ki : cmd_numeric_types = 0.0, 
                 kd : cmd_numeric_types = 10, 
                 effort_limit : cmd_numeric_types = 100.0, 
                 mode : cmd_numeric_types = 3):
        print("[bridge] starting")
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print("[bridge] connected")

        self.publisher = roslibpy.Topic(
            self.client,
            '/xhand_control/xhand_command',
            'xhand_control_ros/XHandCommand'
        )

        self.hand_id = hand_id

        self.kp = XHandPublisher.__convert_to_command_json_type(kp, float)
        self.ki = XHandPublisher.__convert_to_command_json_type(ki, float)
        self.kd = XHandPublisher.__convert_to_command_json_type(kd, float)

        self.effort_limit = XHandPublisher.__convert_to_command_json_type(effort_limit, float)
        self.mode = XHandPublisher.__convert_to_command_json_type(mode, int)

    def send(self, joint_angles):
        if not self.client.is_connected:
            return
        if len(joint_angles) != len(XHandPublisher.JOINT_NAMES):
            return
        
        msg = {
            'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'hand_id':      self.hand_id,
            'name':         XHandPublisher.JOINT_NAMES,
            'position':     joint_angles,
            'kp':           self.kp,
            'ki':           self.ki,
            'kd':           self.kd,
            'effort_limit': self.effort_limit,
            'mode':         self.mode,
        }
        self.publisher.publish(roslibpy.Message(msg))

    def shutdown(self):
        self.publisher.unadvertise()
        self.client.terminate()