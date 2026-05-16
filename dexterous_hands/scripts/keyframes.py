import rospy
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import (
    XHandCommand,
    XHandCommandMsg,
    XHandState,
    XHandStateArrayMsg,
    JOINTS,
    JOINT_LIMITS_RAD,
)
from xhand_control_ros.srv import (
    ResetSensor
)

force_calc = np.zeros((5,3))
def _state_callback(state_msg):
    global force_calc
    state = XHandState.from_msg(state_msg)
    force_calc = np.array(state.sensor.force_calc)
    return

keyframes = {
    "wide_grasp": [
        (0.5, [1.8, -1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        (1, [ 1.9116205, 0.2, 0.65, -0.08264622, 1.03503704, 1.15, 1.10772228, 1.15, 1.04666674, 1.25, 0.9231019, 1.6])
    ],
    "grasp": [
        (0.5, [ 1.9217962, -1.10160193, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] ),
        (0.5, [ 1.87818515, 0.50734264, -0.11048149, -0.11654725, 1.63511119, 0.40497222, 1.65132408, 0.72042598, 1.68771303, 0.78293521, 1.75806491, 0.75002776]), 
    ],
    "pinch": [
        (0.5, [1.8, -1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        (0.5, [ 1.51185179, 1.01468527, -0.02035185, -0.04526527, 1.12807405, 0.98415744, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ],
    "tight_pinch": [
        (0.5, [1.8, -1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        (0.5, [ 1.51185179, 1.21468527, -0.02035185, -0.04526527, 1.12807405, 0.98415744, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ]
}
THRESHOLD = 15

parser = argparse.ArgumentParser(description="XHand keyframe controller")
parser.add_argument(
    "keyframe",
    choices=list(keyframes.keys()),
    help="Name of the keyframe sequence to execute"
)
parser.add_argument(
    "--force","-f", 
    type=float, 
    nargs=1, 
    default=15,
    help="Force threshold for movement cutoff"
)
args = parser.parse_args()

rospy.init_node('xhand_test_controller')
cmd_publisher = rospy.Publisher("/xhand_control/xhand_command", XHandCommandMsg, queue_size=1)
rospy.loginfo("Waiting for subscriber to connect...")
while cmd_publisher.get_num_connections() == 0:
    rospy.sleep(0.2)

rospy.Subscriber("/xhand_control/xhand_state", XHandStateArrayMsg, _state_callback)
    
kp = np.array([25]*12)
kd = np.array([12000]*12)
ki = np.ones_like(kp) * 0.00
eff_limit = np.ones_like(kp) * 250

reset_cmd = XHandCommand(mode=np.array([3]*12),hand_id=0)
reset_cmd.position = np.array([0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
reset_cmd.kp = kp
reset_cmd.ki = ki
reset_cmd.kd = kd
reset_cmd.effort_limit = eff_limit

cmd_publisher.publish(XHandCommand.to_msg(reset_cmd))

rospy.loginfo("Waiting for sensor reset serivce...")
rospy.wait_for_service("/xhand_control/reset_hand_sensor")
rospy.loginfo("Press [ENTER] to reset sensors")
input()

reset_sensors = rospy.ServiceProxy("/xhand_control/reset_hand_sensor", ResetSensor)
try:
    for i in range(17,22):
        resp = reset_sensors(0, i)
        if not resp.success:
            print(resp)
            sys.exit()
except rospy.ServiceException as e:
    print("Service did not process request: " + str(e))


rospy.loginfo("Press [ENTER] to start")
input()

move_cmd = XHandCommand(mode=np.array([3]*12),hand_id=0,kp=np.array([150]*12),ki=ki,kd=kd,effort_limit=eff_limit)
move_cmd.position = reset_cmd.position.copy()

cur_pos = reset_cmd.position.copy()
enabled_joints = np.full((12,),True)
for time, target in keyframes[args.keyframe]:
    t = 0
    target = np.array(target)
    while t < time:
        # Force thresholding checks
        if enabled_joints[0] and force_calc[0,2] > THRESHOLD: # Thumb
            print(f"Froze thumb ({force_calc[0,2]})")
            enabled_joints[0:3] = False
        for i in range(1,5): # Fingers only
            if not enabled_joints[2*i+2]:
                continue

            if force_calc[i,2] > THRESHOLD:
                print(f"Froze {i} ({force_calc[i,2]})")
                enabled_joints[2*i+2:2*i+4] = False

        new_pos = cur_pos + ((target - cur_pos) * (t/time))
        move_cmd.position[enabled_joints] = new_pos[enabled_joints]
        cmd_publisher.publish(XHandCommand.to_msg(move_cmd))

        rospy.sleep(0.01)
        t += 0.01

    cur_pos = target.copy()


rospy.loginfo("Press [ENTER] to reset")
input()

cmd_publisher.publish(XHandCommand.to_msg(reset_cmd))