import rospy
import numpy as np

import sys, os
sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import (
    XHandCommand,
    XHandCommandMsg,
    JOINTS,
    JOINT_LIMITS_RAD,
)

rospy.init_node('xhand_test_controller')
cmd_publisher = rospy.Publisher("/xhand_control/xhand_command", XHandCommandMsg, queue_size=1)
while cmd_publisher.get_num_connections() == 0:
    rospy.loginfo("Waiting for subscriber to connect...")
    rospy.sleep(0.2)

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

rospy.sleep(1)
rospy.loginfo("Running joint test sequence")

move_cmd = XHandCommand(mode=np.array([3]*12),hand_id=0,kp=np.array([100]*12),ki=ki,kd=kd,effort_limit=eff_limit)
move_cmd.position = reset_cmd.position.copy()
for i in range(4):
  move_cmd.position[4 + 2*i] = JOINT_LIMITS_RAD[JOINTS[4 + 2*i]][1]
  move_cmd.position[4 + 2*i + 1] = JOINT_LIMITS_RAD[JOINTS[4 + 2*i + 1]][1]
  
  cmd_publisher.publish(XHandCommand.to_msg(move_cmd))
  rospy.sleep(0.25)
for i in range(4):
  move_cmd.position[4 + 2*i] = JOINT_LIMITS_RAD[JOINTS[4 + 2*i]][0]
  move_cmd.position[4 + 2*i + 1] = JOINT_LIMITS_RAD[JOINTS[4 + 2*i + 1]][0]
  
  cmd_publisher.publish(XHandCommand.to_msg(move_cmd))
  rospy.sleep(0.25)

move_cmd.position[0] = JOINT_LIMITS_RAD[JOINTS[0]][1]
move_cmd.position[1] = JOINT_LIMITS_RAD[JOINTS[1]][1]
cmd_publisher.publish(XHandCommand.to_msg(move_cmd))
rospy.sleep(0.5)
move_cmd.position[1] = 1.0
move_cmd.position[2] = JOINT_LIMITS_RAD[JOINTS[2]][1]
cmd_publisher.publish(XHandCommand.to_msg(move_cmd))
rospy.sleep(0.5)
cmd_publisher.publish(XHandCommand.to_msg(reset_cmd))