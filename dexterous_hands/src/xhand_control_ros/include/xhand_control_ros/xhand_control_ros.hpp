#ifndef __XHAND_CONTROL_ROS_HPP__
#define __XHAND_CONTROL_ROS_HPP__

#include <ros/ros.h>
#include <unistd.h>
#include "xhand_control.hpp"
#include "xhand_control_ros/FingerSensorState.h"
#include "xhand_control_ros/ReadHandInfo.h"
#include "xhand_control_ros/ResetSensor.h"
#include "xhand_control_ros/SetHandId.h"
#include "xhand_control_ros/SetHandName.h"
#include "xhand_control_ros/XHandCommand.h"
#include "xhand_control_ros/XHandSensorState.h"
#include "xhand_control_ros/XHandState.h"
#include "xhand_control_ros/XHandStateArray.h"

namespace xhand_control_ros {
class XHandControlROS {
 public:
  XHandControlROS(ros::NodeHandle& nh);
  ~XHandControlROS();
  bool init();
  void run();
  void calculate_idle_hand_state(); 

 private:
  bool init_parameters();
  void command_callback(const xhand_control_ros::XHandCommand::ConstPtr& msg);
  bool read_hand_info(xhand_control_ros::ReadHandInfo::Request& req,
                      xhand_control_ros::ReadHandInfo::Response& res);
  bool set_hand_id(xhand_control_ros::SetHandId::Request& req,
                   xhand_control_ros::SetHandId::Response& res);
  bool set_hand_name(xhand_control_ros::SetHandName::Request& req,
                     xhand_control_ros::SetHandName::Response& res);
  bool reset_sensor(xhand_control_ros::ResetSensor::Request& req,
                    xhand_control_ros::ResetSensor::Response& res);
  bool map_to_vector(const std::vector<std::string> msg_names,
                     std::vector<int>& map_vector);
  bool is_valid_hand_id(const uint8_t hand_id) const;

 private:
  ros::NodeHandle nh_;
  ros::Publisher state_pub_;
  ros::Subscriber command_sub_;
  ros::ServiceServer read_info_srv_;
  ros::ServiceServer set_id_srv_;
  ros::ServiceServer set_name_srv_;
  ros::ServiceServer reset_sensor_srv_;
  std::shared_ptr<xhand_control::XHandControl> xhand_control_;
  std::vector<uint8_t> hand_ids_;
  HandCommand_t hand_command_;
  HandState_t hand_state_;
  xhand_control_ros::XHandStateArray hand_state_array_;
  xhand_control_ros::XHandStateArray raw_hand_state_array_;   // 初始时读取传感器数值，作为raw state
  std::vector<std::string> finger_joint_names_{
      "thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2",
      "index_bend_joint", "index_joint1",      "index_joint2",
      "mid_joint1",       "mid_joint2",        "ring_joint1",
      "ring_joint2",      "pinky_joint1",      "pinky_joint2"};
  std::vector<double> default_kp_;
  std::vector<double> default_kd_;
  std::vector<double> default_ki_;
  std::vector<double> default_effort_limit_;
  int32_t default_mode_{0};
  double update_rate_{10.0};
};
}  // namespace xhand_control_ros

#endif
