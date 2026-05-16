#ifndef _SERIAL_COMMUNICATION_HPP_
#define _SERIAL_COMMUNICATION_HPP_
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "communiation_interface.hpp"
#include "parse_data.h"
#include "serial/serial.h"

namespace serial_communication {
class SerialCommunication
    : public communication_interface::CommunicationInterface {
 public:
  SerialCommunication(const std::string& port_name, int baud_rate,
                      uint8_t device_id = 0);
  SerialCommunication() = default;
  ~SerialCommunication();
  virtual std::vector<std::string> enumerate_devices() override;
  bool open_device();
  virtual std::vector<uint8_t> list_hands_id();

  virtual DeviceInfo_t read_device_info(uint8_t device_id) override;

  virtual bool send_command(const uint8_t device_id,
                            const HandCommand_t& command) override;

  virtual HandState_t read_state(uint8_t device_id) override;

  virtual HandParam_t read_joint_parameters(uint8_t device_id) override;

  virtual bool set_joint_parameters(const uint8_t device_id,
                                    const HandParam_t& parameters) override;

  virtual bool upgrade_device(const uint8_t device_id,
                              const uint8_t component_id,
                              const std::string& firmware_path) override;

  // virtual bool calibrate_joint(uint8_t device_id) override;

  virtual bool reset_sensor(const uint8_t device_id,
                            const uint8_t sensor_id) override;
  virtual void close_device() override;
  virtual std::string get_firmware_version(const uint8_t device_id,
                                           const uint8_t component_id) override;
  virtual char get_hand_type(uint8_t device_id) override;
  virtual std::string get_serial_number(uint8_t device_id) override;
  virtual bool set_hand_name(const uint8_t device_id,
                             const std::string& name) override;
  virtual std::string get_hand_name(uint8_t device_id) override;
  virtual bool set_hand_id(uint8_t old_id, uint8_t new_id) override;
  // virtual uint8_t get_hand_id(uint8_t device_id) override;
  virtual bool set_hand_parameters(const uint8_t device_id,
                                   HandParam_t& parameters) override;
  virtual HandParam_t get_hand_parameters(uint8_t device_id) override;

 private:
  void parse_callback();
  bool package_and_write(uint8_t target, uint8_t cmd,
                         const std::vector<uint8_t>& data, uint16_t mask = 0x80,
                         uint32_t timeout_ms = 100);
  bool package_and_write(uint8_t target, uint8_t cmd, const uint8_t* data,
                         const uint32_t len, uint16_t mask = 0x80,
                         uint32_t timeout_ms = 100);

 private:
  std::shared_ptr<serial::Serial> serial_;
  std::mutex mutex_;
  std::shared_ptr<std::thread> read_thread_;
  DeviceInfo_t device_info_;
  Parse_Struct_t parse_usb_cdc_;
  uint8_t device_id_{0};
  HandState_t hand_state_;
  SerialHandState_t up_state_;
  std::atomic<bool> exit_{false};
  std::atomic<int> receive_flag_{0};
  std::string joint_version_;
};
}  // namespace serial_communication

#endif  // _SERIAL_COMMUNICATION_HPP_