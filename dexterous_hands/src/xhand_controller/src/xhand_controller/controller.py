from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
import rospy
from nptyping import Int, NDArray
from std_srvs.srv import SetBool as SetBoolSrv
from std_srvs.srv import SetBoolRequest as SetBoolSrvReq
from std_srvs.srv import SetBoolResponse as SetBoolSrvRes

from .utils import Smoother

if TYPE_CHECKING:
    # Python LSP needs import from the root directory for the incompletion with ROS environment
    from xhand_utils.src.xhand_utils import (
        EFFORT_LIMIT,
        JOINTS,
        NUM_JOINTS,
        XHandCommand,
        XHandCommandMsg,
        XHandState,
        XHandStateArrayMsg,
    )
else:
    # if ROS environment is activated, the following can be imported
    from xhand_utils import (
        EFFORT_LIMIT,
        JOINTS,
        NUM_JOINTS,
        XHandCommand,
        XHandCommandMsg,
        XHandState,
        XHandStateArrayMsg,
    )

JOINT_TYPE = Union[Sequence[str], Sequence[int], np.ndarray]  # joints
CONTROL_PARAM_TYPE = Union[np.ndarray, None]  # control parameters


class ControllerBase:
    """
    A base class for controlling joints in a robotic hand.

    Attributes:
        _control_joints (NDArray[Any, Int]): An array of joint indices to be controlled.
        _log_data (dict): A dictionary to store log data.

    Methods:
        __call__(state: XHandState, cmd: XHandCommand = XHandCommand(), **params) -> XHandCommand:
            Calls the pre_forward method and then the forward method.

        pre_forward(**params):
            Prepares the controller by setting parameters before the forward pass.

        forward(state: XHandState, cmd: XHandCommand) -> XHandCommand:
            Abstract method to be implemented by subclasses for the forward pass.

        joints_mapping(joints: Union[JOINT_TYPE, None]) -> NDArray[Any, Int]:
            Converts joint names to joint indices.

        control_joints() -> NDArray[Any, Int]:
            Property to get the control joints.

        log(**kwargs):
            Logs the provided keyword arguments.
    """

    def __init__(
        self,
        control_joints: Union[JOINT_TYPE, None],
    ):
        self._control_joints = self.joints_mapping(control_joints)
        self._log_data = {}

    def __call__(self, state: XHandState, cmd: XHandCommand = XHandCommand(), **params) -> XHandCommand:
        """
        Executes the controller logic when the instance is called.

        Args:
            state (XHandState): The current state of the XHand.
            cmd (XHandCommand, optional): The command to be executed. Defaults to a new instance of XHandCommand.
            **params: Additional parameters for the pre_forward method.

        Returns:
            XHandCommand: The resulting command after processing.
        """
        self.pre_forward(**params)
        return self.forward(state, cmd.copy())

    def pre_forward(self, **params):
        """
        Prepares the controller by setting parameters before the forward pass.

        This method iterates over the attributes of the controller instance. If an attribute is an instance of
        ControllerBase, it recursively calls the `pre_forward` method on that attribute. If an attribute's name
        matches a key in the provided parameters, it sets the attribute to the corresponding value from the parameters.

        Parameters:
        -----------
        **params : dict
            A dictionary of parameters to set on the controller. Keys are attribute names and values are the values
            to set for those attributes.
        """
        for key, value in self.__dict__.items():
            # if the value is a controller, set the parameters of the controller recursively
            if isinstance(value, ControllerBase):
                value.pre_forward(**params)
            elif key in params:
                # set the parameters of the controller
                setattr(self, key, params[key])

    def forward(self, state: XHandState, cmd: XHandCommand) -> XHandCommand:
        """
        Processes the given state and command to generate a new command.

        Args:
            state (XHandState): The current state of the XHand.
            cmd (XHandCommand): The command to be processed.

        Returns:
            XHandCommand: The new command generated based on the current state and input command.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def joints_mapping(joints: Union[JOINT_TYPE, None]) -> NDArray[Any, Int]:  # type: ignore
        """
        Convert joint names to joint indices.

        Parameters:
        joints (Union[JOINT_TYPE, None]): A list or array of joint names or indices, or None.

        Returns:
        NDArray[Any, Int]: An array of joint indices. If `joints` is None, returns an array of indices for all joints.
        """

        if joints is None:
            return np.arange(NUM_JOINTS)
        if isinstance(joints, np.ndarray):
            return joints
        return np.array([JOINTS.index(joint) if isinstance(joint, str) else joint for joint in joints])

    @property
    def control_joints(self) -> NDArray[Any, Int]:  # type: ignore
        """
        Controls the joints of the robotic hand.

        Returns:
            NDArray[Any, Int]: An array representing the control values for the joints.
        """
        return self._control_joints

    def log(self, **kwargs):
        """
        Logs the provided keyword arguments.

        This method takes any number of keyword arguments and logs their values.
        If a key does not already exist in the log data, it initializes a list for that key.
        Then, it appends the value to the list corresponding to the key.

        Args:
            **kwargs: Arbitrary keyword arguments to be logged. Each key represents
                      a log category, and the value is the data to be logged under that category.
        """
        for key, value in kwargs.items():
            if key not in self._log_data:
                self._log_data[key] = []
            self._log_data[key].append(value)


class PositionPIDController(ControllerBase):
    """
    A PID controller for managing the position of joints in a robotic hand.

    Attributes:
        kp (CONTROL_PARAM_TYPE): Proportional gain.
        ki (CONTROL_PARAM_TYPE): Integral gain.
        kd (CONTROL_PARAM_TYPE): Derivative gain.
        target_position (CONTROL_PARAM_TYPE): Desired target position for the joint.
        effort_limit (CONTROL_PARAM_TYPE): Maximum effort limit for the joint.
        control_joints (Union[JOINT_TYPE, None]): The joints to be controlled by this PID controller.

    Methods:
        forward(state: XHandState, cmd: XHandCommand) -> XHandCommand:
            Computes the control command based on the current state and the PID parameters.
            Logs the current position, effort, target position, and effort limit if applicable.
    """

    def __init__(
        self,
        kp: CONTROL_PARAM_TYPE,
        ki: CONTROL_PARAM_TYPE,
        kd: CONTROL_PARAM_TYPE,
        target_position: CONTROL_PARAM_TYPE,
        effort_limit: CONTROL_PARAM_TYPE,
        control_joints: Union[JOINT_TYPE, None],
    ):
        super().__init__(control_joints)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_position = target_position
        self.effort_limit = effort_limit

    def forward(self, state: XHandState, cmd: XHandCommand) -> XHandCommand:
        params = {}
        if self.kp is not None:
            params["kp"] = self.kp
        if self.ki is not None:
            params["ki"] = self.ki
        if self.kd is not None:
            params["kd"] = self.kd
        if self.target_position is not None:
            params["position"] = self.target_position
        if self.effort_limit is not None:
            params["effort_limit"] = self.effort_limit

        cmd.masked_set(self.control_joints, **params.copy())

        self.log(
            position=state.joint.position[self.control_joints],
            effort=state.joint.effort[self.control_joints],
            target_position=self.target_position,
        )
        if self.effort_limit is not None:
            self.log(effort_limit=self.effort_limit)

        return cmd


class EffortController(ControllerBase):
    """
    EffortController is a controller class that manages the effort control of joints in a robotic hand.

    Args:
        ke (CONTROL_PARAM_TYPE): Control gain parameter.
        target_effort (CONTROL_PARAM_TYPE): Target effort for the controller.
        control_joints (Union[JOINT_TYPE, None]): Joints to be controlled.
        effort_limit (Union[CONTROL_PARAM_TYPE, None], optional): Limit for the effort. Defaults to None.
        frequency (float): Control frequency.
        effort_smooth (bool, optional): Flag to enable effort smoothing. Defaults to True.
        position_smooth (bool, optional): Flag to enable position smoothing. Defaults to False.
        effort_history_len (int, optional): Length of the effort history for smoothing. Defaults to 5.
        position_history_len (int, optional): Length of the position history for smoothing. Defaults to 5.
        control_type (str, optional): Type of control ("open" or "closed"). Defaults to "open".
        delta_position_limit (Union[CONTROL_PARAM_TYPE, None], optional): Limit for the delta position. Defaults to None.

    Methods:
        forward(state: XHandState, cmd: XHandCommand) -> XHandCommand:
            Computes the control command based on the current state and target effort.

        reset(mask: Union[CONTROL_PARAM_TYPE, None] = None) -> None:
            Resets the delta position, optionally using a mask.
    """

    def __init__(
        self,
        ke: CONTROL_PARAM_TYPE,
        target_effort: CONTROL_PARAM_TYPE,
        control_joints: Union[JOINT_TYPE, None],
        effort_limit: Union[CONTROL_PARAM_TYPE, None] = None,
        *,
        frequency: float,
        effort_smooth: bool = True,
        position_smooth: bool = False,
        effort_history_len: int = 5,
        position_history_len: int = 5,
        control_type: str = "open",
        delta_position_limit: Union[CONTROL_PARAM_TYPE, None] = None,
    ) -> None:
        super().__init__(control_joints)
        self.frequency = frequency

        if effort_smooth:
            self.effort_smoother = Smoother(
                T=1 / self.frequency,
                alpha=0.1,
                beta=0.01,
                history_len=effort_history_len,
            )
        else:
            self.effort_smoother = None
        if position_smooth:
            self.position_smoother = Smoother(
                T=1 / self.frequency,
                alpha=0.1,
                beta=0.01,
                history_len=position_history_len,
            )
        else:
            self.position_smoother = None

        self.ke = ke
        self.effort_limit = effort_limit
        self.target_effort = target_effort

        self._delta_position = np.zeros_like(self.ke)

        self._control_type = control_type

        self._delta_position_limit = delta_position_limit

    def forward(self, state: XHandState, cmd: XHandCommand) -> XHandCommand:
        if self.target_effort is None:
            raise ValueError("target_effort is not set")
        effort = state.joint.effort[self.control_joints].copy()
        self.log(effort=effort)

        if self.effort_smoother is not None:
            effort_filtered = self.effort_smoother(effort)
            self.log(effort_filtered=effort_filtered)
            effort = effort_filtered
        effort_error = self.target_effort - effort

        if self._control_type == "closed":
            pos = cmd.position[self.control_joints].copy()
            delta_pos_current = self.ke * effort_error
            self._delta_position += delta_pos_current
        elif self._control_type == "open":
            pos = state.joint.position[self.control_joints].copy()
            delta_pos_current = self.ke * effort_error
            self._delta_position = delta_pos_current
        else:
            raise ValueError(f"Invalid control type: {self._control_type}")

        if self._delta_position_limit is not None:
            clipped_delta_position = np.clip(
                self._delta_position, -self._delta_position_limit, self._delta_position_limit
            )
            self.log(delta_position_clipped=clipped_delta_position)

        position_reference = pos + (
            self._delta_position if self._delta_position_limit is None else clipped_delta_position
        )
        if self.position_smoother is not None:
            position_reference_smoothed = self.position_smoother(position_reference)
            self.log(position_reference_smoothed=position_reference_smoothed)

        self.log(
            delta_pos_current=delta_pos_current,
            position=state.joint.position[self.control_joints],
            position_0=pos,
            position_reference=position_reference,
            delta_position=self._delta_position,
            effort_error=effort_error,
            target_effort=self.target_effort,
            effort_limit=self.effort_limit,
        )
        # print(f"effort_error:{effort_error},pos{pos},delta_pos_current{delta_pos_current}\nself._delta_position{self._delta_position},position_reference{position_reference}")
        params = {}
        params["position"] = position_reference if self.position_smoother is None else position_reference_smoothed
        if self.effort_limit is not None:
            params["effort_limit"] = self.effort_limit

        cmd.masked_set(self.control_joints, **params.copy())

        return cmd

    def reset(self, mask: Union[CONTROL_PARAM_TYPE, None] = None) -> None:
        self._delta_position = self._delta_position * (mask if mask is not None else 0)


class SwitchedController(ControllerBase):
    """
    A controller that switches between position control and effort control.

    Attributes:
        position_controller (PositionPIDController): The position PID controller.
        effort_controller (EffortController): The effort controller.
        kt (Union[CONTROL_PARAM_TYPE, None]): The control parameter type or None.
        ros_service (Union[str, None]): The ROS service name or None.
        default_enable (bool): The default state of effort control.

    Methods:
        toggle_effort_control_cbk(req: SetBoolSrvReq) -> SetBoolSrvRes:
            Callback to toggle effort control via ROS service.

        toggle_effort_control(enable: bool) -> None:
            Toggle effort control manually.

        forward(state: XHandState, cmd: XHandCommand) -> XHandCommand:
            Compute the control command based on the current state and command.

        enable_effort_control() -> np.ndarray:
            Get the current state of effort control.

        enable_effort_control(value: Union[bool, np.ndarray]) -> None:
            Set the state of effort control.
    """

    def __init__(
        self,
        position_controller: PositionPIDController,
        effort_controller: EffortController,
        kt: Union[CONTROL_PARAM_TYPE, None] = None,
        *,
        ros_service: Union[str, None] = None,
        default_enable: bool = False,
    ) -> None:
        super().__init__(np.union1d(position_controller.control_joints, effort_controller.control_joints))
        self.position_controller = position_controller
        self.effort_controller = effort_controller
        self.kt = kt
        #print(f"kt{kt}")
        self.enable_effort_control = default_enable

        if ros_service is not None:
            self.ros_service = rospy.Service(ros_service, SetBoolSrv, self.toggle_effort_control_cbk)
        else:
            self.ros_service = None

        self.delta_position = np.zeros_like(self.effort_controller.control_joints, dtype=float)

    def toggle_effort_control_cbk(self, req: SetBoolSrvReq) -> SetBoolSrvRes:
        self.enable_effort_control = bool(req.data)

        return SetBoolSrvRes(success=True)

    def toggle_effort_control(self, enable: bool) -> None:
        self.enable_effort_control = enable

    def forward(self, state: XHandState, cmd: XHandCommand) -> XHandCommand:
        control_joints = self.effort_controller.control_joints

        # position controller
        cmd = self.position_controller(state, cmd)
        #print(f"positioncmd{cmd.position}")
        # print(f"位控计算cmd {cmd}")
        self.log(position_controller=cmd.position[control_joints])
        self.log(enable_effort_control=int(self.enable_effort_control))

        # effort controller
        target_delta_position = np.zeros_like(control_joints, dtype=float)
        if self.enable_effort_control:
            target_delta_position = (self.effort_controller(state, cmd).position - cmd.position)[control_joints]
        else:
            target_delta_position = np.zeros_like(control_joints, dtype=float)
            self.effort_controller.reset()
        #print(f"target_delta_position:{target_delta_position}")
        self.log(target_delta_position=target_delta_position)
        # print(f"力控计算cmd {cmd}")
        # transit process
        if self.kt is not None:
            # sprint(f"kt is not None{kt}")
            self.delta_position += self.kt * (target_delta_position - self.delta_position)
        else:
            self.delta_position = target_delta_position
        self.log(delta_position=self.delta_position)

        # Final control
        target_position = cmd.position[control_joints] + self.delta_position
        self.log(target_position=target_position)
        #print(f"delta_postion:{self.delta_position}")
        #print(f"target_postion:{target_position}")
        cmd.masked_set(control_joints, position=target_position)
        #print(f"maskcmd:{cmd.position}")
        return cmd


def initialize_system(
    *,
    kp: CONTROL_PARAM_TYPE = np.ones(NUM_JOINTS) * 20,
    ki: CONTROL_PARAM_TYPE = np.ones(NUM_JOINTS) * 0.0,
    kd: CONTROL_PARAM_TYPE = np.ones(NUM_JOINTS) * 0.5,
    target_position: CONTROL_PARAM_TYPE = np.ones(NUM_JOINTS) * 0.2,
    effort_limit: CONTROL_PARAM_TYPE = np.ones(NUM_JOINTS) * EFFORT_LIMIT[-1],
    control_joints: Union[JOINT_TYPE, None] = None,
    state_topic: str = "/xhand_control/xhand_state",
    command_topic: str = "/xhand_control/xhand_command",
    length: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        rospy.init_node("DataCollector")
    except rospy.ROSException:
        pass

    cmd = XHandCommand()
    control_joints = ControllerBase.joints_mapping(control_joints)
    # 新增修改初始化后的关节位置
    # target_position = np.array([1.57, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0])
    cmd.masked_set(indices=control_joints, kp=kp, ki=ki, kd=kd, position=target_position, effort_limit=effort_limit)
    cmd.effort_limit = np.ones_like(target_position) * 100 # 新增代码为了减小初始化时的动静
    rospy.loginfo("Resetting the system...")

    for i in range(30):
        rospy.wait_for_message(state_topic, XHandStateArrayMsg, timeout=5)
        publisher = rospy.Publisher(command_topic, XHandCommandMsg, queue_size=1)
        publisher.publish(XHandCommand.to_msg(cmd))
        #rospy.sleep(0.01) # new
        # print(f"初始化命令：{XHandCommand.to_msg(cmd)}")

    rospy.sleep(3.0)

    rospy.loginfo("System reset complete, collecting data...")

    states: List[XHandState] = []
    for i in range(length):
        state_msg = rospy.wait_for_message(state_topic, XHandStateArrayMsg, timeout=5)
        states.append(XHandState.from_msg(state_msg))  # type: ignore
    efforts = np.array([state.joint.effort[control_joints] for state in states])

    calc_forces = np.array([state.sensor.force_calc for state in states])

    mean_efforts = np.mean(efforts, axis=0)
    std_efforts = np.std(efforts, axis=0)
    filtered_efforts = efforts[np.all(np.abs(efforts - mean_efforts) <= 2 * std_efforts, axis=1)]
    idle_efforts = np.mean(filtered_efforts, axis=0)

    x = calc_forces[..., 0]
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    filtered_x = x[np.all(np.abs(x - mean_x) <= 2 * std_x, axis=1)]
    idle_x = np.mean(filtered_x, axis=0)

    y = calc_forces[..., 1]
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    filtered_y = y[np.all(np.abs(y - mean_y) <= 2 * std_y, axis=1)]
    idle_y = np.mean(filtered_y, axis=0)

    z = calc_forces[..., 2]
    mean_z = np.mean(z, axis=0)
    std_z = np.std(z, axis=0)
    filtered_z = z[np.all(np.abs(z - mean_z) <= 2 * std_z, axis=1)]
    idle_z = np.mean(filtered_z, axis=0)
    return idle_efforts, np.vstack((idle_x, idle_y, idle_z)).swapaxes(0, 1)
