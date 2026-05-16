import numpy as np
import roslibpy
import sys, os

# --- EMA Smoother ---

class EMASmoother:
    def __init__(self, alpha=0.2, num_joints=12):
        self.alpha = alpha
        self.state = None

    def update(self, new_angles):
        if self.state is None:
            self.state = np.array(new_angles, dtype=float)
        else:
            self.state = self.alpha * np.array(new_angles) + (1 - self.alpha) * self.state
        return self.state.copy()


# --- Joint Names ---

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

# --- Joint Limits (radians) ---

JOINT_LIMITS = {
    "thumb_bend_joint":  (0.0,   1.57),
    "thumb_rota_joint1": (-1.05, 1.57),
    "thumb_rota_joint2": (0.0,   1.57),
    "index_bend_joint":  (-0.09, 0.3),
    "index_joint1":      (0.0,   1.92),
    "index_joint2":      (0.0,   1.92),
    "mid_joint1":        (0.0,   1.92),
    "mid_joint2":        (0.0,   1.92),
    "ring_joint1":       (0.0,   1.92),
    "ring_joint2":       (0.0,   1.92),
    "pinky_joint1":      (0.0,   1.92),
    "pinky_joint2":      (0.0,   1.92),
}


# --- Angle Calculations ---

def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def signed_angle_between(a, b, c, reference_normal):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    cross = np.cross(ba, bc)
    if np.dot(cross, reference_normal) < 0:
        angle = -angle
    return angle


def palm_normal(lm):
    palm_vec1 = np.array(lm[5])  - np.array(lm[17])
    palm_vec2 = np.array(lm[0])  - np.array(lm[9])
    normal = np.cross(palm_vec1, palm_vec2)
    normal /= (np.linalg.norm(normal) + 1e-6)
    return normal


def thumb_cmc_rotation(lm, normal):
    return signed_angle_between(lm[5], lm[0], lm[1], normal)

def index_abduction_angle(lm, normal):
    return signed_angle_between(lm[9], lm[0], lm[5], normal)

def flexion_angle(a, b, c):
    """
    Returns 0 when straight, increases as finger bends.
    """
    return np.pi - angle_between(a, b, c)

def landmarks_to_angles(hand_landmarks):
    lm = [(l.x, l.y, l.z) for l in hand_landmarks]
    normal = palm_normal(lm)

    return [
        thumb_cmc_rotation(lm, normal),
        flexion_angle(lm[1], lm[2], lm[3]),        # thumb_rota_joint1
        flexion_angle(lm[2], lm[3], lm[4]),        # thumb_rota_joint2

        index_abduction_angle(lm, normal),
        flexion_angle(lm[0], lm[5], lm[6]),        # index_joint1
        flexion_angle(lm[5], lm[6], lm[7]),        # index_joint2

        flexion_angle(lm[0], lm[9],  lm[10]),      # mid_joint1
        flexion_angle(lm[9], lm[10], lm[11]),      # mid_joint2

        flexion_angle(lm[0], lm[13], lm[14]),      # ring_joint1
        flexion_angle(lm[13],lm[14], lm[15]),      # ring_joint2

        flexion_angle(lm[0], lm[17], lm[18]),      # pinky_joint1
        flexion_angle(lm[17],lm[18], lm[19]),      # pinky_joint2
    ]


def clamp_angles(angles):
    return [
        float(np.clip(angle, lo, hi))
        for angle, (lo, hi) in zip(angles, JOINT_LIMITS.values())
    ]


# --- roslibpy Publisher ---

class XHandPublisher:
    def __init__(self, host='localhost', port=9090, hand_id=0,
                 kp=10.0, ki=0.0, kd=0.5, effort_limit=10.0, mode=1):
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()

        self.publisher = roslibpy.Topic(
            self.client,
            '/xhand_control/xhand_command',
            'xhand_control_ros/XHandCommand'
        )

        self.smoother     = EMASmoother(alpha=0.2, num_joints=12)
        self.hand_id      = hand_id
        n = len(JOINT_NAMES)
        self.kp           = [kp]           * n
        self.ki           = [ki]           * n
        self.kd           = [kd]           * n
        self.effort_limit = [effort_limit] * n
        self.mode         = [mode]         * n

    def send(self, hand_landmarks):
        if not self.client.is_connected:
            return

        raw      = landmarks_to_angles(hand_landmarks)
        smoothed = self.smoother.update(raw)
        clamped  = clamp_angles(smoothed)

        msg = {
            'header': {
                'stamp': {'secs': 0, 'nsecs': 0},
                'frame_id': ''
            },
            'hand_id': self.hand_id,
            'name':         JOINT_NAMES,
            'position':     clamped,
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