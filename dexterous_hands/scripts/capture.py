import rospy
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time

sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import XHandState, XHandStateArrayMsg

captured = False

def _state_callback(state_msg):
    global captured
    if captured:
        return
    state = XHandState.from_msg(state_msg)
    print(state.joint.position)
    captured = True

rospy.init_node('xhand_state_feedback')
rospy.Subscriber("/xhand_control/xhand_state", XHandStateArrayMsg, _state_callback)

while not captured:
    time.sleep(0.1)
