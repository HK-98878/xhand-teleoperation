# Reset joints to a base position, then disengage (mode 0) to allow free movement

import rospy
import numpy as np

import sys, os
sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import (
    XHandCommand,
    XHandCommandMsg,
)

rospy.init_node('xhand_reset')
cmd_publisher = rospy.Publisher("/xhand_control/xhand_command", XHandCommandMsg, queue_size=1)
while cmd_publisher.get_num_connections() == 0:
    rospy.loginfo("Waiting for subscriber to connect...")
    rospy.sleep(1)

rospy.loginfo("Resetting joints")
reset_cmd = XHandCommand(mode=np.array([3]*12),hand_id=0)
reset_cmd.position = np.array([0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
reset_cmd.kp = np.array([25]*12)
reset_cmd.ki = np.array([12000]*12)
reset_cmd.kd = np.array([0]*12)
reset_cmd.effort_limit = np.array([250]*12)

cmd_publisher.publish(XHandCommand.to_msg(reset_cmd))

rospy.sleep(1)
rospy.loginfo("Disengaging joints")
reset_cmd.mode = np.array([0]*12)
cmd_publisher.publish(XHandCommand.to_msg(reset_cmd))