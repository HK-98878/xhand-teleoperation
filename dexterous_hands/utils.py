import rosbag
import numpy as np
import matplotlib.pyplot as plt

def visuzalize_rosbag_data(bag: rosbag.Bag):
    pos_epi = []
    eff_epi = []
    cf_epi = []
    target_pos_epi = []
    for topic, msg, t in bag.read_messages():
        if topic == "/hand_control/cachestate":
            pos = []
            efforts = []
            calc_forces = []
            for i in range(len(msg.hand_id)):
                joint_positions = np.asarray(msg.hand_states[i].position, dtype=np.float32)
                joint_efforts = np.asarray(msg.hand_states[i].effort, dtype=np.float32)
                pos.append(joint_positions)
                efforts.append(joint_efforts)
                for sensor in msg.sensor_states[i].finger_sensor_states:
                    calc_force = sensor.calc_force
                    calc_forces.append(np.asarray([calc_force.x, calc_force.y, calc_force.z]))
            pos = np.concatenate(pos)
            efforts = np.concatenate(efforts)
            calc_forces = np.stack(calc_forces, axis=0).flatten()
            pos_epi.append(pos)
            eff_epi.append(efforts)
            cf_epi.append(calc_forces)
        elif topic == "/xhand_control/xhand_command":
            positions = np.asarray(msg.position, dtype=np.float32)
            target_pos_epi.append(positions)
        else:
            raise NotImplementedError("Unknown topic")
    time_steps = np.arange(len(pos_epi))
    cf_epi = np.asarray(cf_epi)

    plt.figure(figsize=(10, 6))
    finger_name = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for i in range(0, 5):
        plt.plot(time_steps, cf_epi[:, i * 3 + 0], label=finger_name[i])
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Visualization of Tactile X Force')
    plt.legend()
    plt.grid(True)
    plt.show()
    for i in range(0, 5):
        plt.plot(time_steps, cf_epi[:, i * 3 + 1], label=finger_name[i])
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Visualization of Tactile Y Force')
    plt.legend()
    plt.grid(True)
    plt.show()
    for i in range(0, 5):
        plt.plot(time_steps, cf_epi[:, i * 3 + 2], label=finger_name[i])
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Visualization of Tactile Z Force')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bag_file = "hand16.bag"
    bag = rosbag.Bag(bag_file)
    visuzalize_rosbag_data(bag)
