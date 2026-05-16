#include "xhand_control_ros/xhand_control_ros.hpp"
#include <typeinfo>
namespace xhand_control_ros {

XHandControlROS::XHandControlROS(ros::NodeHandle& nh) : nh_(nh) {}

XHandControlROS::~XHandControlROS() {}

bool XHandControlROS::init() {
  if (!init_parameters()) {
    ROS_ERROR("Failed to initialize parameters");
    return false;
  }
  xhand_control_ = std::make_unique<xhand_control::XHandControl>();
  auto ifnames = xhand_control_->enumerate_devices("EtherCAT");
  if (ifnames.empty()) {
    ROS_ERROR("No XHand devices found");
    return false;
  }
  std::string device_name{ifnames[0]};
  if (xhand_control_->open_ethercat(device_name)) {
    ROS_INFO("XHand device opened");
    hand_ids_ = xhand_control_->list_hands_id();
    if (hand_ids_.empty()) {
      ROS_ERROR("No XHand devices found");
      return false;
    }
    std::vector<std::string> finger{"thumb", "index", "middle", "ring",
                                    "pinky"};
    for (auto& id : hand_ids_) {
      // Hand state
      hand_state_array_.hand_id.push_back(id);
      hand_state_array_.hand_name.push_back(xhand_control_->get_hand_name(id));
      hand_state_array_.hand_type.push_back(
          std::string(1, xhand_control_->get_hand_type(id)));
      xhand_control_ros::XHandState state;
      state.name = finger_joint_names_;
      state.position.resize(finger_joint_names_.size(), 0);
      state.effort.resize(finger_joint_names_.size(), 0);
      state.temperature.resize(finger_joint_names_.size(), 0);
      state.error_code.resize(finger_joint_names_.size(), 0);
      hand_state_array_.hand_states.push_back(state);

      // sensor state
      xhand_control_ros::XHandSensorState sensor_state;
      for (int i = 0; i < finger.size(); i++) {
        xhand_control_ros::FingerSensorState finger_sensor_state;
        finger_sensor_state.location = finger[i];
        finger_sensor_state.raw_force.resize(120);
        finger_sensor_state.raw_temperature.resize(20);
        sensor_state.finger_sensor_states.push_back(finger_sensor_state);
      }
      hand_state_array_.sensor_states.push_back(sensor_state);
      // hand_sensor_state_array_.hand_name.push_back
    }

    // new start
    // 初始化 raw_hand_state_array_
    for (auto& id : hand_ids_) {
      // Hand state
      raw_hand_state_array_.hand_id.push_back(id);
      raw_hand_state_array_.hand_name.push_back(xhand_control_->get_hand_name(id));
      raw_hand_state_array_.hand_type.push_back(
          std::string(1, xhand_control_->get_hand_type(id)));

      xhand_control_ros::XHandState state;
      state.name = finger_joint_names_;
      state.position.resize(finger_joint_names_.size(), 0);
      state.effort.resize(finger_joint_names_.size(), 0);
      state.temperature.resize(finger_joint_names_.size(), 0);
      state.error_code.resize(finger_joint_names_.size(), 0);
      raw_hand_state_array_.hand_states.push_back(state);

      // Sensor state
      xhand_control_ros::XHandSensorState sensor_state;
      for (int i = 0; i < finger.size(); i++) {
        xhand_control_ros::FingerSensorState finger_sensor_state;
        finger_sensor_state.location = finger[i];
        finger_sensor_state.raw_force.resize(120);
        finger_sensor_state.raw_temperature.resize(20);
        sensor_state.finger_sensor_states.push_back(finger_sensor_state);
      }
      raw_hand_state_array_.sensor_states.push_back(sensor_state);
    }

    // 打印 raw_hand_state_array_
    // ROS_INFO_STREAM("Raw Hand State Array Initialized: \n" << raw_hand_state_array_);
    //end

    state_pub_ =
        nh_.advertise<xhand_control_ros::XHandStateArray>("xhand_state", 1);
    command_sub_ = nh_.subscribe("xhand_command", 1,
                                 &XHandControlROS::command_callback, this);
    read_info_srv_ = nh_.advertiseService(
        "read_hand_info", &XHandControlROS::read_hand_info, this);
    set_id_srv_ = nh_.advertiseService("set_hand_id",
                                       &XHandControlROS::set_hand_id, this);
    set_name_srv_ = nh_.advertiseService("set_hand_name",
                                         &XHandControlROS::set_hand_name, this);
    reset_sensor_srv_ = nh_.advertiseService(
        "reset_hand_sensor", &XHandControlROS::reset_sensor, this);
    return true;
  }
  return false;
}

void XHandControlROS::run() {
  ros::Rate update_rate(update_rate_);
  while (ros::ok()) {
    for (auto& id : hand_ids_) {
      // std::cout<<"---"<<id<<"***"<<std::endl;
      hand_state_ = xhand_control_->read_state(id);
      for (int i = 0; i < hand_state_array_.hand_id.size(); i++) {
        if (hand_state_array_.hand_id[i] == id) {
          auto& pub_state = hand_state_array_.hand_states[i];
          for (int joint_idx = 0; joint_idx < hand_state_.finger_state.size();
               joint_idx++) {
            pub_state.position[joint_idx] =
                hand_state_.finger_state[joint_idx].position;
            pub_state.effort[joint_idx] = hand_state_.finger_state[joint_idx].torque; 
            int16_t effort_ = static_cast<int16_t>(pub_state.effort[joint_idx]);
            pub_state.effort[joint_idx] = static_cast<int16_t>(hand_state_.finger_state[joint_idx].torque); 
            // pub_state.effort[joint_idx] -= raw_hand_state_array_.hand_states[i].effort[joint_idx];  // -offset hand_switch_controller不要idle;
            std::cout<<"***"<<static_cast<uint16_t>(hand_state_.finger_state[joint_idx].default5)<<"---"<< effort_ <<"---"<<pub_state.effort[joint_idx]<<"---"<<raw_hand_state_array_.hand_states[i].effort[joint_idx]<<"---"<<std::endl;
          }

          for (int j = 0; j < hand_state_.senser_data.size(); j++) {
            auto& read_sensor_state = hand_state_.senser_data[j];
            auto& pub_sensor_state =
                hand_state_array_.sensor_states[i].finger_sensor_states[j];
            /*pub_sensor_state.calc_force.x = read_sensor_state.calc_force.fx - raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.x; // -offset
            pub_sensor_state.calc_force.y = read_sensor_state.calc_force.fy - raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.y; // -offset
            pub_sensor_state.calc_force.z = read_sensor_state.calc_force.fz - raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.z; // -offset*/
            
            pub_sensor_state.calc_force.x = read_sensor_state.calc_force.fx;
            pub_sensor_state.calc_force.y = read_sensor_state.calc_force.fy;
            pub_sensor_state.calc_force.z = read_sensor_state.calc_force.fz;
            pub_sensor_state.calc_force.x -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.x;
            pub_sensor_state.calc_force.y -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.y;
            pub_sensor_state.calc_force.z -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.z;
            std::cout<<"***"<<pub_sensor_state.calc_force.z<<"---"<<typeid(read_sensor_state.calc_force.fz).name()<<"---"<<raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].calc_force.z<<"---"<<std::endl;
            for (int k = 0; k < read_sensor_state.raw_force.size(); k++) {
              pub_sensor_state.raw_force[k].x =
                  read_sensor_state.raw_force[k].fx;
              pub_sensor_state.raw_force[k].y =
                  read_sensor_state.raw_force[k].fy;
              pub_sensor_state.raw_force[k].z =
                  read_sensor_state.raw_force[k].fz;
              pub_sensor_state.raw_force[k].x -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].raw_force[k].x;
              pub_sensor_state.raw_force[k].y -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].raw_force[k].y;
              pub_sensor_state.raw_force[k].z -= raw_hand_state_array_.sensor_states[i].finger_sensor_states[j].raw_force[k].z;
            }
            pub_sensor_state.calc_temperature =
                read_sensor_state.calc_temperature;
            for (int k = 0; k < read_sensor_state.temperature.size(); k++) {
              pub_sensor_state.raw_temperature[k] =
                  read_sensor_state.temperature[k];
            }
          }
          break;
        }
      }
    }
    hand_state_array_.header.stamp = ros::Time::now();
    state_pub_.publish(hand_state_array_);
    // ROS_INFO_STREAM("Rcollected_states: \n" << hand_state_array_);
    ros::spinOnce();
    update_rate.sleep();
  }
}

void XHandControlROS::calculate_idle_hand_state() {
  for (int i = 0; i < 500; i++) {
    // 创建一个与 raw_hand_state_array_ 结构相同的临时变量
    // xhand_control_ros::XHandStateArray temp_raw_hand_state_array = raw_hand_state_array_;
    for (auto& id : hand_ids_) {
      hand_state_ = xhand_control_->read_state(id);
      for (int j = 0; j < raw_hand_state_array_.hand_id.size(); j++) {
        if (raw_hand_state_array_.hand_id[j] == id) {
          // 赋值手指关节状态
          auto& pub_state = raw_hand_state_array_.hand_states[j];
          for (int joint_idx = 0; joint_idx < hand_state_.finger_state.size(); joint_idx++) {
            pub_state.position[joint_idx] = hand_state_.finger_state[joint_idx].position;
            //pub_state.effort[joint_idx] += hand_state_.finger_state[joint_idx].torque;
            pub_state.effort[joint_idx] += static_cast<int16_t>(hand_state_.finger_state[joint_idx].torque);
          }
          // 赋值传感器状态
          for (int k = 0; k < hand_state_.senser_data.size(); k++) {
            auto& read_sensor_state = hand_state_.senser_data[k];
            auto& pub_sensor_state = raw_hand_state_array_.sensor_states[j].finger_sensor_states[k];

            pub_sensor_state.calc_force.x += read_sensor_state.calc_force.fx;
            pub_sensor_state.calc_force.y += read_sensor_state.calc_force.fy;
            pub_sensor_state.calc_force.z += read_sensor_state.calc_force.fz;

            // 赋值原始力数据
            for (int m = 0; m < read_sensor_state.raw_force.size(); m++) {
              pub_sensor_state.raw_force[m].x += read_sensor_state.raw_force[m].fx;
              pub_sensor_state.raw_force[m].y += read_sensor_state.raw_force[m].fy;
              pub_sensor_state.raw_force[m].z += read_sensor_state.raw_force[m].fz;
            }

            // 赋值温度数据
            pub_sensor_state.calc_temperature = read_sensor_state.calc_temperature;
            for (int m = 0; m < read_sensor_state.temperature.size(); m++) {
              pub_sensor_state.raw_temperature[m] = read_sensor_state.temperature[m];
            }
          }
          break;
        }
      }
      usleep(20000);
      // sleep(0.01);
    }

    //ROS_INFO_STREAM("Rcollected_states: \n" << raw_hand_state_array_);
    //ROS_INFO_STREAM("Rcollected_states: \n" << collected_states);
    //ROS_INFO("print end");
  }
  // ROS_INFO_STREAM("Rcollected_states: \n" << raw_hand_state_array_);
  ROS_INFO("print end1");

  // 计算均值
  for (int j = 0; j < raw_hand_state_array_.hand_id.size(); j++) {
    auto& pub_state = raw_hand_state_array_.hand_states[j];
    for (int joint_idx = 0; joint_idx < pub_state.effort.size(); joint_idx++) {
      pub_state.effort[joint_idx] /= 500.0;
    }
    for (int k = 0; k < raw_hand_state_array_.sensor_states[j].finger_sensor_states.size(); k++) {
      auto& pub_sensor_state = raw_hand_state_array_.sensor_states[j].finger_sensor_states[k];

      pub_sensor_state.calc_force.x /= 500.0;
      pub_sensor_state.calc_force.y /= 500.0;
      pub_sensor_state.calc_force.z /= 500.0;
      // std::cout<<pub_sensor_state.calc_force.z<<std::endl;
      for (int m = 0; m < pub_sensor_state.raw_force.size(); m++) {
        pub_sensor_state.raw_force[m].x /= 500.0;
        pub_sensor_state.raw_force[m].y /= 500.0;
        pub_sensor_state.raw_force[m].z /= 500.0;
      }
    }
  }
  //ROS_INFO_STREAM("Rcollected_states: \n" << raw_hand_state_array_);
  ROS_INFO("print end2");
}

bool XHandControlROS::init_parameters() {
  int error = 0;
  error += !nh_.getParam("update_rate", update_rate_);
  // error += !nh_.getParam("kp", default_kp_);
  // error += !nh_.getParam("kd", default_kd_);
  // error += !nh_.getParam("ki", default_ki_);
  // error += !nh_.getParam("effort_limit", default_effort_limit_);
  // error += !nh_.getParam("mode", default_mode_);

  return error == 0;
}

void XHandControlROS::command_callback(
    const xhand_control_ros::XHandCommand::ConstPtr& msg) {
  if (!is_valid_hand_id(msg->hand_id)) {
    // print(msg->hand_id)
    // std::cout<<"command"<<msg->hand_id<<"comamnd"<<std::endl;
    ROS_ERROR("Invalid hand id");
    return;
  }
  if (msg->name.size() != 0 && msg->name.size() == msg->position.size() &&
      msg->name.size() == msg->kp.size() &&
      msg->name.size() == msg->kd.size() &&
      msg->name.size() == msg->ki.size() &&
      msg->name.size() == msg->effort_limit.size() &&
      msg->name.size() == msg->mode.size()) {
    std::vector<int> map_vec;
    if (!map_to_vector(msg->name, map_vec)) {
      ROS_ERROR("Invalid command");
      return;
    }
    HandCommand_t cmd;
    for (int i = 0; i < cmd.finger_command.size(); ++i) {
      cmd.finger_command[i].id = i;
      cmd.finger_command[i].position = 0;
      cmd.finger_command[i].kp = 0;
      cmd.finger_command[i].kd = 0;
      cmd.finger_command[i].ki = 0;
      cmd.finger_command[i].tor_max = 0;
      cmd.finger_command[i].mode = 0;
    }
    for (int i = 0; i < msg->name.size(); i++) {
      cmd.finger_command[map_vec[i]].id = map_vec[i];
      cmd.finger_command[map_vec[i]].position = msg->position[i];
      cmd.finger_command[map_vec[i]].kp = msg->kp[i];
      cmd.finger_command[map_vec[i]].kd = msg->kd[i];
      cmd.finger_command[map_vec[i]].ki = msg->ki[i];
      cmd.finger_command[map_vec[i]].tor_max = msg->effort_limit[i];
      //cmd.finger_command[map_vec[i]].mode = msg->mode;
      cmd.finger_command[map_vec[i]].mode = msg->mode[i];
    }
    xhand_control_->send_command(msg->hand_id, cmd);
  } else {
    ROS_ERROR("Invalid command");
  }
}
bool XHandControlROS::read_hand_info(
    xhand_control_ros::ReadHandInfo::Request& req,
    xhand_control_ros::ReadHandInfo::Response& res) {
  if (!is_valid_hand_id(req.hand_id)) {
    res.success = false;
    res.info = "Invalid hand id";
    return true;
  }

  switch (req.info_type) {
    case xhand_control_ros::ReadHandInfo::Request::HAND_TPYE:
      res.success = true;
      res.info = std::string(1, xhand_control_->get_hand_type(req.hand_id));
      break;
    case xhand_control_ros::ReadHandInfo::Request::SERIAL_NUMBER:
      res.success = true;
      res.info = xhand_control_->get_serial_number(req.hand_id);
      break;
    case xhand_control_ros::ReadHandInfo::Request::VERSION:
      res.success = true;
      res.info = xhand_control_->read_version(req.hand_id);
      break;
    case xhand_control_ros::ReadHandInfo::Request::HAND_NAME:
      res.success = true;
      res.info = xhand_control_->get_hand_name(req.hand_id);
      break;
    default:
      res.success = false;
      res.info = "Invalid info type";
      break;
  }
  return true;
}
bool XHandControlROS::set_hand_id(xhand_control_ros::SetHandId::Request& req,
                                  xhand_control_ros::SetHandId::Response& res) {
  if (!is_valid_hand_id(req.current_id)) {
    res.success = false;
    res.message = "Invalid hand id";
    return true;
  }
  res.success = xhand_control_->set_hand_id(req.current_id, req.new_id);
  return true;
}
bool XHandControlROS::set_hand_name(
    xhand_control_ros::SetHandName::Request& req,
    xhand_control_ros::SetHandName::Response& res) {
  if (!is_valid_hand_id(req.hand_id)) {
    res.success = false;
    res.message = "Invalid hand id";
    return true;
  }
  res.success = xhand_control_->set_hand_name(req.hand_id, req.hand_name);
  return true;
}
bool XHandControlROS::reset_sensor(
    xhand_control_ros::ResetSensor::Request& req,
    xhand_control_ros::ResetSensor::Response& res) {
  if (!is_valid_hand_id(req.hand_id)) {
    res.success = false;
    res.message = "Invalid hand id";
    return true;
  }
  if (req.sensor_id < 0x11 || req.sensor_id > 0x15) {
    res.success = false;
    res.message = "Invalid sensor id";
    return true;
  }
  res.success = xhand_control_->reset_sensor(req.hand_id, req.sensor_id);

  return true;
}
bool XHandControlROS::map_to_vector(const std::vector<std::string> msg_names,
                                    std::vector<int>& map_vector) {
  map_vector.resize(msg_names.size());
  for (int i = 0; i < msg_names.size(); i++) {
    bool found = false;
    for (int j = 0; j < finger_joint_names_.size(); j++) {
      if (finger_joint_names_[j] == msg_names[i]) {
        map_vector[i] = j;
        found = true;
        break;
      }
    }
    if (!found) {
      ROS_ERROR("Invalid finger name: %s", msg_names[i].c_str());
      return false;
    }
  }
  return true;
}

bool XHandControlROS::is_valid_hand_id(const uint8_t hand_id) const {
  if (std::find(hand_ids_.begin(), hand_ids_.end(), hand_id) ==
      hand_ids_.end()) {
    return false;
  }
  return true;
}
}  // namespace xhand_control_ros