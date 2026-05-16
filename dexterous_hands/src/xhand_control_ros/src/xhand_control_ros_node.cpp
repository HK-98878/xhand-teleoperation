#include "xhand_control_ros/xhand_control_ros.hpp"

int main(int argc, char** argv) {
  using namespace xhand_control_ros;
  ros::init(argc, argv, "xhand_control");
  ros::NodeHandle nh("xhand_control");
  XHandControlROS xhand_control(nh);
  if (!xhand_control.init()) {
    ROS_ERROR("Failed to initialize xhand_control");
    // return 0;
  } else {
    xhand_control.calculate_idle_hand_state();
    usleep(10000);
    xhand_control.run();
  }
  ros::shutdown();

  return 0;
}