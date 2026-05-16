from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from typing import Any, List, Tuple, TypeVar

import geometry_msgs.msg as geo_msgs
import numpy as np
import pandas as pd
#import rosbag
import rospy
from nptyping import NDArray, Shape

from xhand_control_ros.msg import (
    FingerSensorState as FingerSensorStateMsg,
)
from xhand_control_ros.msg import (
    XHandCommand as XHandCommandMsg,
)
from xhand_control_ros.msg import (
    XHandSensorState as XHandSensorStateMsg,
)
from xhand_control_ros.msg import (
    XHandState as XHandStateMsg,
)
from xhand_control_ros.msg import (
    XHandStateArray as XHandStateArrayMsg,
)

#      /-2--       <-- thumb bend joint
#     /1          <-- thumb rota joint1
#  --0         <-- thumb rota joint2
#     \3-4---5--    <-- index bend joint, index joint1, index joint2
#     |--6---7--    <-- mid joint1, mid joint2
#     |--8---9--    <-- ring joint1, ring joint2
#     |--10--11-   <-- pinky joint1, pinky joint2

# joint names,from inner to outer
JOINTS = [
    "thumb_bend_joint",  # 0
    "thumb_rota_joint1",  # 1
    "thumb_rota_joint2",  # 2
    "index_bend_joint",  # 3
    "index_joint1",  # 4
    "index_joint2",  # 5
    "mid_joint1",  # 6
    "mid_joint2",  # 7
    "ring_joint1",  # 8
    "ring_joint2",  # 9
    "pinky_joint1",  # 10
    "pinky_joint2",  # 11
]

# 手指名称
FINGERS = ["thumb", "index", "mid", "ring", "pinky"]

# 关节限制
JOINT_LIMITS_RAD = {
    "thumb_bend_joint": (0, 1.57),
    "thumb_rota_joint1": (-1.05, 1.57),
    "thumb_rota_joint2": (0, 1.57),
    "index_bend_joint": (-0.09, 0.3),
    "index_joint1": (0, 1.92),
    "index_joint2": (0, 1.92),
    "mid_joint1": (0, 1.92),
    "mid_joint2": (0, 1.92),
    "ring_joint1": (0, 1.92),
    "ring_joint2": (0, 1.92),
    "pinky_joint1": (0, 1.92),
    "pinky_joint2": (0, 1.92),
}

# 力限制
EFFORT_LIMIT = (0, 400)

# 一些常量
NUM_JOINTS = 12  # <-- 12个关节
NUM_FINGERS = 5  # <-- 5个手指
NUM_SENSOR_ROWS = 12  # <-- 每个手指有12行传感器
NUM_SENSOR_COLS = 10  # <-- 每个手指有10列传感器
NUM_SENSOR_DEPTH = 3  # <-- 每个传感器有3个维度(x, y, z)

# 类型提示
JOINT_TYPE = NDArray[Shape[f"{NUM_JOINTS}"], Any]  # <-- 关节类型，形状为(12, )
FINGER_TYPE = NDArray[Shape[f"{NUM_FINGERS}"], Any]  # <-- 手指类型，形状为(5, )

FINGER_FORCE_CALC_TYPE = NDArray[  # <-- 手指力类型，形状为(5, 3)
    Shape[f"{NUM_FINGERS}, {NUM_SENSOR_DEPTH}"], Any
]
FORCE_TYPE = NDArray[  # <-- 单个手指力类型，形状为(12, 10, 3)
    Shape[f"{NUM_SENSOR_ROWS}, {NUM_SENSOR_COLS}, {NUM_SENSOR_DEPTH}"],
    Any,
]
FINGER_FORCE_TYPE = NDArray[  # <-- 手指原始力类型，形状为(5, 12, 10, 3)
    Shape[f"{NUM_FINGERS}, {NUM_SENSOR_ROWS}, {NUM_SENSOR_COLS}, {NUM_SENSOR_DEPTH}"],
    Any,
]
T = TypeVar("T", bound="XHandBase")


@dataclass
class XHandBase:
    """
    A base class for handling operations related to XHand.

    Methods
    -------
    from_msg(msg: Any) -> "XHandBase"
        Static method to create an instance of XHandBase from a ros message.

    to_msg(data: "XHandBase") -> Any
        Static method to convert an instance of XHandBase to a ros message.

    _masked_set(indices: Any, data: np.ndarray, attribute: str) -> None
        Protected method to set data at specified indices for a given attribute.

    masked_set(indices: Any, **kwargs) -> None
        Method to set data at specified indices for multiple attributes.
    """

    @staticmethod
    def from_msg(msg: Any) -> "XHandBase":
        raise NotImplementedError

    @staticmethod
    def to_msg(data: "XHandBase") -> Any:
        raise NotImplementedError

    def _masked_set(self, indices: Any, data: np.ndarray, attribute: str) -> None:
        if hasattr(self, attribute):
            getattr(self, attribute)[indices] = data

    def masked_set(self, indices: Any, **kwargs) -> None:
        for attribute, data in kwargs.items():
            self._masked_set(indices, data, attribute)

    def copy(self: T) -> T:
        copied_dict = {
            key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in self.__dict__.items()
        }
        return self.__class__(**copied_dict)


@dataclass
class XHandJoint(XHandBase):
    """
    Represents the state of a joint in the XHand robotic system.

    Attributes:
        position (JOINT_TYPE): The position of the joint.
        effort (JOINT_TYPE): The effort exerted by the joint.
        temperature (JOINT_TYPE): The temperature of the joint.
        error_code (JOINT_TYPE): The error code of the joint, initialized to zeros.

    Methods:
        from_msg(msg: XHandStateMsg) -> "XHandJoint":
            Creates an XHandJoint instance from an XHandState ros message.

        to_msg(joint_state: "XHandJoint") -> XHandStateMsg:
            Converts an XHandJoint instance to an XHandState ros message.
    """

    position: JOINT_TYPE = np.zeros(NUM_JOINTS)
    effort: JOINT_TYPE = np.zeros(NUM_JOINTS)
    temperature: JOINT_TYPE = np.zeros(NUM_JOINTS)
    error_code: JOINT_TYPE = np.zeros(NUM_JOINTS)

    @staticmethod
    def from_msg(msg: XHandStateMsg) -> "XHandJoint":
        pos = np.array(msg.position).reshape(NUM_JOINTS)
        effort = np.array(msg.effort).reshape(NUM_JOINTS)
        temperature = np.array(msg.temperature).reshape(NUM_JOINTS)
        error_code = np.array(msg.error_code).reshape(NUM_JOINTS)

        return XHandJoint(
            position=pos,
            effort=effort,
            temperature=temperature,
            error_code=error_code,
        )

    @staticmethod
    def to_msg(joint_state: "XHandJoint") -> XHandStateMsg:
        msg = XHandStateMsg()

        msg.name = JOINTS
        msg.position = joint_state.position.tolist()
        msg.effort = joint_state.effort.tolist()
        msg.temperature = joint_state.temperature.tolist()
        msg.error_code = joint_state.error_code.tolist()

        return msg


@dataclass
class XHandSensor:
    """
    A class to represent the sensor data of an XHand.

    Attributes:
    -----------
    force_calc : FINGER_FORCE_CALC_TYPE
        Calculated forces for each finger.
    force_raw : FINGER_FORCE_TYPE
        Raw forces for each finger.

    Methods:
    --------
    from_msg(msg: XHandSensorStateMsg) -> "XHandSensor":
        Converts a message of type XHandSensorState ros msg to an XHandSensor object.

    to_msg(sensor_state: "XHandSensor") -> XHandSensorStateMsg:
        Converts an XHandSensor object to a message of type XHandSensorState ros msg.
    """

    force_calc: FINGER_FORCE_CALC_TYPE = np.zeros((NUM_FINGERS, NUM_SENSOR_DEPTH))
    force_raw: FINGER_FORCE_TYPE = np.zeros((NUM_FINGERS, NUM_SENSOR_ROWS, NUM_SENSOR_COLS, NUM_SENSOR_DEPTH))
    # TODO: calc temp & raw temp

    @staticmethod
    def from_msg(msg: XHandSensorStateMsg) -> "XHandSensor":
        calc_forces = []
        raw_forces = []

        for finger_msg in msg.finger_sensor_states:  # type:ignore
            finger: FingerSensorStateMsg = finger_msg
            calc_forces.append(
                [
                    finger.calc_force.x,
                    finger.calc_force.y,
                    finger.calc_force.z,
                ]
            )

            raw_finger_forces = []

            for force in finger.raw_force:  # type:ignore
                raw_finger_forces.append([force.x, force.y, force.z])

            raw_forces.append(raw_finger_forces)

        calc_force = np.array(calc_forces).reshape(NUM_FINGERS, NUM_SENSOR_DEPTH)
        raw_force = np.array(raw_forces).reshape(
            NUM_FINGERS,
            NUM_SENSOR_ROWS,
            NUM_SENSOR_COLS,
            NUM_SENSOR_DEPTH,
        )

        return XHandSensor(force_calc=calc_force, force_raw=raw_force)

    @staticmethod
    def to_msg(sensor_state: "XHandSensor") -> XHandSensorStateMsg:
        msg = XHandSensorStateMsg()

        msg.finger_sensor_states = []

        for i in range(NUM_FINGERS):
            finger_msg = FingerSensorStateMsg()

            finger_msg.calc_force.x, finger_msg.calc_force.y, finger_msg.calc_force.z = sensor_state.force_calc[i]

            raw_forces = []
            for j, f in enumerate(
                sensor_state.force_raw.reshape(
                    NUM_FINGERS,
                    NUM_SENSOR_ROWS * NUM_SENSOR_COLS,
                    NUM_SENSOR_DEPTH,
                )[i]
            ):
                raw_force = geo_msgs.Vector3()
                raw_force.x, raw_force.y, raw_force.z = f
                raw_forces.append(raw_force)

            finger_msg.raw_force = raw_forces
            msg.finger_sensor_states.append(finger_msg)

        return msg


@dataclass
class XHandState(XHandBase):
    """
    Represents the state of an XHand, including joint and sensor states,
    identification, and timestamp information.

    Attributes:
        joint (XHandJoint): The joint state of the XHand.
        sensor (XHandSensor): The sensor state of the XHand.
        hand_id (int): The ID of the hand. Default is 0.
        hand_name (str): The name of the hand. Default is "0".
        hand_type (str): The type of the hand. Default is "0".
        frame_id (str): The frame ID associated with the hand state.
        time_stamp (float): The timestamp of the hand state.

    Methods:
        from_msg(msg: XHandStateArrayMsg) -> "XHandState":
            Creates an XHandState instance from a given XHandStateArray ros message.

        to_msg(hand_state: "XHandState") -> XHandStateArrayMsg:
            Converts an XHandState instance to an XHandStateArray ros message.
    """

    joint: XHandJoint = XHandJoint()
    sensor: XHandSensor = XHandSensor()
    hand_id: int = 0
    hand_name: str = "0"
    hand_type: str = "0"
    frame_id: str = ""
    time: float = 0.0

    @staticmethod
    def from_msg(msg: XHandStateArrayMsg) -> "XHandState":
        joint_state = XHandJoint.from_msg(msg.hand_states[0])  # type:ignore
        sensor_state = XHandSensor.from_msg(msg.sensor_states[0])  # type:ignore
        return XHandState(
            joint=joint_state,
            sensor=sensor_state,
            hand_id=msg.hand_id[0],
            hand_name=msg.hand_name[0],
            hand_type=msg.hand_type[0],
            frame_id=msg.header.frame_id,
            time=msg.header.stamp.to_time(),
        )

    @staticmethod
    def to_msg(hand_state: "XHandState") -> XHandStateArrayMsg:
        msg = XHandStateArrayMsg()

        msg.header.frame_id = hand_state.frame_id
        msg.header.stamp = rospy.Time.from_sec(hand_state.time)

        msg.hand_id = [hand_state.hand_id]
        msg.hand_name = [hand_state.hand_name]
        msg.hand_type = [hand_state.hand_type]

        msg.hand_states = [XHandJoint.to_msg(hand_state.joint)]
        msg.sensor_states = [XHandSensor.to_msg(hand_state.sensor)]
        return msg


@dataclass
class XHandCommand(XHandBase):
    """
    XHandCommand represents a command to control the XHand robotic hand.

    Attributes:
        position (JOINT_TYPE): The target positions for each joint.
        kp (JOINT_TYPE): The proportional gain for each joint.
        ki (JOINT_TYPE): The integral gain for each joint.
        kd (JOINT_TYPE): The derivative gain for each joint.
        effort_limit (JOINT_TYPE): The effort limits for each joint.
        mode (int): The control mode (0: position control, 1: force control, 2: velocity control, 3: position control).
        hand_id (int): The identifier for the hand.

    Methods:
        from_msg(msg: XHandCommandMsg) -> "XHandCommand":
            Creates an XHandCommand instance from an XHandCommand ros message.

        to_msg(cmd: "XHandCommand") -> XHandCommandMsg:
            Converts an XHandCommand instance to an XHandCommand ros message.
    """

    position: JOINT_TYPE = np.zeros(NUM_JOINTS)
    kp: JOINT_TYPE = np.zeros(NUM_JOINTS)
    ki: JOINT_TYPE = np.zeros(NUM_JOINTS)
    kd: JOINT_TYPE = np.zeros(NUM_JOINTS)
    effort_limit: JOINT_TYPE = np.ones(NUM_JOINTS) * EFFORT_LIMIT[-1]
    # mode: int = 3  # <-- 0: 零力矩, 1: , 2: , 3: 位置控制
    mode : JOINT_TYPE = np.ones(NUM_JOINTS,dtype=int) * 3 # ros更新了，可以同时用不同的模式控制不同的关节np.ones_like(pre_grasp_kp, dtype=int) * 3
    hand_id: int = 0
    time: float = 0.0

    @staticmethod
    def from_msg(msg: XHandCommandMsg) -> "XHandCommand":
        position = np.array(msg.position).reshape(NUM_JOINTS)
        kp = np.array(msg.kp).reshape(NUM_JOINTS)
        ki = np.array(msg.ki).reshape(NUM_JOINTS)
        kd = np.array(msg.kd).reshape(NUM_JOINTS)
        effort_limit = np.array(msg.effort_limit).reshape(NUM_JOINTS)

        return XHandCommand(
            hand_id=msg.hand_id,
            position=position,
            kp=kp,
            ki=ki,
            kd=kd,
            effort_limit=effort_limit,
            mode=msg.mode,
            time=msg.header.stamp.to_time(),
        )

    @staticmethod
    def to_msg(cmd: "XHandCommand") -> XHandCommandMsg:
        cmd_msg = XHandCommandMsg()
        cmd_msg.header.stamp = rospy.Time.now()

        cmd_msg.hand_id = cmd.hand_id
        cmd_msg.name = JOINTS
        cmd_msg.position = cmd.position.tolist()
        cmd_msg.kp = cmd.kp.tolist()
        cmd_msg.ki = cmd.ki.tolist()
        cmd_msg.kd = cmd.kd.tolist()
        cmd_msg.effort_limit = cmd.effort_limit.tolist()
        # cmd_msg.mode = cmd.mode
        cmd_msg.mode = cmd.mode.tolist() # 更新

        return cmd_msg


'''class XHandDataCollector:
    def __init__(self) -> None: ...
    def collect(self, method: str = "offline", **kwargs) -> pd.DataFrame:
        topics = kwargs.get("topics", [])
        if method == "offline":
            if "rosbag" not in kwargs:
                raise ValueError("Rosbag file must be provided")
            rosbag_file = kwargs.get("rosbag", None)
            msgs = self._offline_collector(rosbag_file, topics)

        elif method == "online":
            duration = kwargs.get("duration", 10.0)
            msgs = self._online_collector(duration, topics)

        else:
            raise ValueError("Invalid method. Must be either 'offline' or 'online'")
        df = self._to_dataframe(msgs)
        if "kind" in kwargs and kwargs["kind"] == "nice":
            df.drop(
                columns=[
                    col
                    for col in df.columns
                    if "sensor" in col
                    or "mode" in col
                    or "hand_id" in col
                    or "hand_name" in col
                    or "hand_type" in col
                    or "error" in col
                    or "frame" in col
                    or "temperature" in col
                ],
                inplace=True,
            )
        return df

    def _offline_collector(self, rosbag_file, topics) -> "dict[str,list]":
        bag = rosbag.Bag(rosbag_file)
        msgs = {}
        for msg in bag.read_messages():
            topic_name = msg[0]
            if topic_name not in topics:
                continue

            topic_time = msg[-1].to_time()
            raw_msg = msg[1]

            wrapped_msg = _wrap_msg(raw_msg, time=topic_time)
            if topic_name not in msgs:
                msgs[topic_name] = []
            msgs[topic_name].append(wrapped_msg)
        return msgs

    def _online_collector(self, duration: float, topics: List[Tuple[str, Any]]) -> "dict[str, list]":
        """
        In order to collect data from the topics, we need to create a callback class
        Maybe there is a better way to do this, but I don't know yet
        """

        class _Callback:
            def __init__(self):
                self.msgs = {}
                self._cnt = 0

            def callbacks(self, msg):
                topic_name = msg._connection_header["topic"]
                topic_type = msg._connection_header["type"].split("/")[-1]
                try:
                    deserialized_msg = globals()[topic_type + "Msg"]()
                except Exception as e:
                    rospy.logerr(f"Error: {e}")
                    return
                deserialized_msg.deserialize(msg._buff)

                topic_time = rospy.Time.now().to_time()

                wrapped_msg = _wrap_msg(deserialized_msg, time=topic_time)
                if topic_name not in self.msgs:
                    self.msgs[topic_name] = []
                self.msgs[topic_name].append(wrapped_msg)
                self._cnt += 1

        callback = _Callback()

        # if node is not initialized, we need to initialize it
        try:
            rospy.init_node("DataCollector")
        except rospy.ROSException:
            pass
        rospy.loginfo("Collecting data...")

        for topic in topics:
            # topic_name, topic_type = topic
            rospy.Subscriber(topic, rospy.AnyMsg, callback.callbacks)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed_time = current_time - start_time
            # 检查是否超过指定的持续时间
            if elapsed_time.to_sec() > duration:
                rospy.loginfo(f"Finished collecting data for {duration} seconds, Total messages: {callback._cnt}")
                break
            # 睡眠一段时间以避免占用过多的CPU
            rospy.sleep(1)
        return callback.msgs

    def _to_dataframe(self, msgs) -> pd.DataFrame:
        data_frames = []
        for topic, data in msgs.items():
            topic_df = pd.DataFrame([_flatten_dict_and_array(asdict(msg)) for msg in data])
            topic_df.columns = [f"{topic}_{col}" if col != "time" else col for col in topic_df.columns]
            data_frames.append(topic_df)

        merged_df = pd.merge(*data_frames, on="time", how="outer")
        merged_df["time"] = merged_df["time"] - merged_df["time"][0]

        return merged_df'''


def _get_msg_type(msg):
    return msg._type.split("/")[-1]


def _wrap_msg(msg, **kwargs):
    msg_type = _get_msg_type(msg)

    if msg_type == "XHandStateArray":
        msg = XHandState.from_msg(msg)
    elif msg_type == "XHandCommand":
        msg = XHandCommand.from_msg(msg)
    else:
        raise ValueError("Invalid message type")
    msg.time = kwargs.get("time", 0.0)

    return msg


def _flatten_dict_and_array(d: MutableMapping, parent_key: str = "", sep: str = ".") -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict_and_array(v, new_key, sep=sep).items())
        elif isinstance(v, np.ndarray) and not isinstance(v, str):
            v = v.flatten().tolist()
            for i, u in enumerate(v):
                items.extend(_flatten_dict_and_array({str(i): u}, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
