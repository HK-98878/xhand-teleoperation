"""
ROS publisher via roslibpy (websocket -> rosbridge_server).

The script runs OUTSIDE the ROS container; on the ROS side you need:
    rosrun rosbridge_server rosbridge_websocket
or in ROS 2:
    ros2 launch rosbridge_server rosbridge_websocket_launch.xml

Topic format
------------
We publish std_msgs/Float64MultiArray on the configured topic. The data array
is the 12-element joint vector in the order defined by XHAND1_JOINT_NAMES.
The MultiArrayLayout's `dim[0].label` is set to "xhand1_joints" and the labels
are encoded into `layout.dim[0].size` slots so the consumer can verify ordering.

If you'd rather use sensor_msgs/JointState, the swap is one line in `publish()`.
"""
from __future__ import annotations
import threading
import time
from typing import Optional

import numpy as np

try:
    import roslibpy  # type: ignore
    _HAVE_ROSLIBPY = True
except ImportError:
    _HAVE_ROSLIBPY = False

from .retarget_xhand1 import XHAND1_JOINT_NAMES


class RosBridgePublisher:
    """Pluggable ROS publisher. If `enabled=False`, every call is a no-op."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        topic: str = "/xhand1/target_joint_positions",
        msg_type: str = "std_msgs/Float64MultiArray",
        enabled: bool = True,
        use_jointstate: bool = False,
    ):
        self.enabled = enabled
        self.host = host
        self.port = port
        self.topic_name = topic
        self.use_jointstate = use_jointstate
        self.msg_type = "sensor_msgs/JointState" if use_jointstate else msg_type
        self._client: Optional[object] = None
        self._topic: Optional[object] = None
        self._connected = False
        self._lock = threading.Lock()

    def start(self):
        if not self.enabled:
            print("[ros] disabled")
            return
        if not _HAVE_ROSLIBPY:
            raise RuntimeError("roslibpy not installed. `pip install roslibpy` "
                               "or run with --no-ros.")
        self._client = roslibpy.Ros(host=self.host, port=self.port)
        self._client.on_ready(self._on_connect)
        self._client.run()
        # roslibpy.run() is non-blocking; give it a moment to connect.
        for _ in range(20):
            if self._connected:
                break
            time.sleep(0.05)
        if not self._connected:
            print(f"[ros] WARNING: not connected to rosbridge at {self.host}:{self.port} "
                  f"(continuing; will keep retrying in background)")

    def _on_connect(self):
        self._topic = roslibpy.Topic(self._client, self.topic_name, self.msg_type)
        self._topic.advertise()
        self._connected = True
        print(f"[ros] connected, advertising {self.topic_name} ({self.msg_type})")

    def stop(self):
        if not self.enabled:
            return
        try:
            if self._topic is not None:
                self._topic.unadvertise()
            if self._client is not None:
                self._client.terminate()
        except Exception:
            pass

    def publish(self, joint_positions: np.ndarray):
        if not self.enabled or not self._connected or self._topic is None:
            return
        joints = np.asarray(joint_positions, dtype=np.float64).tolist()
        if self.use_jointstate:
            msg = {
                "header": {"stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
                "name": list(XHAND1_JOINT_NAMES),
                "position": joints,
                "velocity": [],
                "effort": [],
            }
        else:
            msg = {
                "layout": {
                    "dim": [{
                        "label": "xhand1_joints",
                        "size": len(joints),
                        "stride": len(joints),
                    }],
                    "data_offset": 0,
                },
                "data": joints,
            }
        with self._lock:
            try:
                self._topic.publish(roslibpy.Message(msg))
            except Exception as e:
                # Silent during teleop; you can promote to a counter if wanted.
                pass
