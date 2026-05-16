import rospy
import numpy as np
import sys, os

sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import XHandState, XHandStateArrayMsg

FINGER = 4  # pinky — change to whichever finger you're testing

def _state_callback(state_msg):
    state = XHandState.from_msg(state_msg)
    sensor = state.sensor.force_raw[FINGER].astype(float)
    
    fn = sensor[:, :, 2]  # normal force — most reliable for localisation
    
    if fn.max() > 20:  # threshold to ignore noise — tune as needed
        row, col = np.unravel_index(np.argmax(fn), fn.shape)
        print(f"Peak contact: row={row}, col={col}  (fn={fn.max():.2f})")
        print(fn.astype(int))
        print()

rospy.init_node('xhand_calibration')
rospy.Subscriber("/xhand_control/xhand_state", XHandStateArrayMsg, _state_callback)
rospy.spin()