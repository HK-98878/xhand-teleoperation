#ifndef _XHAND_CONTROL_HPP__
#define _XHAND_CONTROL_HPP__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "communiation_interface.hpp"
#include "visibility_control.h"

using namespace communication_interface;
namespace xhand_control {
class XHandControl {
 public:
  XHandControl() = default;
  ~XHandControl() = default;
  /**
   * @brief  Enumerate devices connected to the computer
   *
   * @param comm_type  Communication type, "serial" or "ethercat"
   * @return std::vector<std::string>  List of device names
   */
  XHAND_CONTROL_PUBLIC std::vector<std::string> enumerate_devices(
      const std::string& comm_type);

  /** @brief  Open a serial device
   *
   * @param port  Serial port name
   * @param baudrate  Baudrate
   * @param device_id  Device ID
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool open_serial(const std::string& port,
                                        uint32_t baudrate);
  /** @brief  Open an ethercat device
   *
   * @param ifname  Ethercat interface name
   * @param device_id  Device ID
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool open_ethercat(const std::string& ifname);
  /**
   * @brief  Get device information
   * @return DeviceInfo_t  Device information
   */
  XHAND_CONTROL_PUBLIC DeviceInfo_t read_device_info(uint8_t device_id);
  /** @brief  Send a command to the device
   *
   * @param command  Command to send
   * @return true  Successcommand
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool send_command(const uint8_t device_id,
                                         const HandCommand_t& command);
  /** @brief  Read the state of the device
   *
   * @return HandState_t  Device state
   */
  XHAND_CONTROL_PUBLIC HandState_t read_state(uint8_t device_id);
  /** @brief  Read the parameters of the device
   *
   * @return HandParam_t  Device parameters
   */
  XHAND_CONTROL_PUBLIC HandParam_t read_parameters(uint8_t device_id);
  /** @brief  Set the parameters of the device
   *
   * @param parameters  Parameters to set
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool set_parameters(const uint8_t device_id,
                                           const HandParam_t& parameters);
  /** @brief  Upgrade the firmware of the device
   *
   * @param firmware_path  Path to the firmware file
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool upgrade_device(const uint8_t device_id,
                                           const uint8_t component_id,
                                           const std::string& firmware_path);
  XHAND_CONTROL_PUBLIC std::string read_version(const uint8_t device_id);
  /** @brief  Calibrate the device
   *
   * @param step  Calibration step
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool calibrate_joint(const uint8_t device_id,
                                            const uint8_t step);
  /** @brief  Reset the sensor of the device
   *
   * @param sensor_id  Sensor ID
   * @return true  Success
   * @return false  Failure
   */
  XHAND_CONTROL_PUBLIC bool reset_sensor(const uint8_t device_id,
                                         const uint8_t sensor_id);
  /** @brief  Close the device
   *
   */
  XHAND_CONTROL_PUBLIC void close_device();

  XHAND_CONTROL_PUBLIC std::vector<uint8_t> list_hands_id();
  XHAND_CONTROL_PUBLIC char get_hand_type(uint8_t device_id);
  XHAND_CONTROL_PUBLIC std::string get_serial_number(uint8_t device_id);
  XHAND_CONTROL_PUBLIC bool set_hand_name(const uint8_t device_id,
                                          const std::string& name);
  XHAND_CONTROL_PUBLIC std::string get_hand_name(uint8_t device_id);
  XHAND_CONTROL_PUBLIC bool set_hand_id(uint8_t old_id, uint8_t new_id);

  // XHAND_CONTROL_PUBLIC bool set_hand_parameters(const uint8_t device_id,
  //                                               HandParam_t& parameters);
  // XHAND_CONTROL_PUBLIC HandParam_t get_hand_parameters(uint8_t device_id);

 private:
  std::shared_ptr<communication_interface::CommunicationInterface> device_;
  HandParam_t hand_parameters_;
  uint8_t calibration_step_{1};
};
}  // namespace xhand_control

#endif  // _XHAND_CONTROL_HPP__