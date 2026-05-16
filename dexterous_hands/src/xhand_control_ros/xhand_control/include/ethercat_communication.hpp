#ifndef _ETHERCAT_COMMUNICATION_HPP_
#define _ETHERCAT_COMMUNICATION_HPP_
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "communiation_interface.hpp"
#include "ethercat.h"
#include "parse_data.h"

namespace ethercat_communication {
using PdModePdoOutputType = FingerCommand_t;

using PdModePdoInputType = UpState_t;

class EthercatCommunication
    : public communication_interface::CommunicationInterface {
 public:
  EthercatCommunication(const std::string& ifname);
  EthercatCommunication() = default;
  ~EthercatCommunication();
  virtual std::vector<std::string> enumerate_devices() override;
  bool open_device();
  virtual std::vector<uint8_t> list_hands_id();

  virtual DeviceInfo_t read_device_info(uint8_t device_id = 0xFF) override;

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
  int sdo_write(uint16_t index, uint8_t subindex, uint8_t* data, uint16_t len);
  int sdo_read(uint16_t index, uint8_t subindex, uint8_t* data, uint16_t len);
  int set_watch_dog(ec_slavet* slave, int milliseconds);
  void ethercat_update();
  void add_timespec(struct timespec* ts, int64 addtime);
  void cycle_delay(struct timespec ts);
  static int servo_setup(uint16 slave);
  std::vector<uint8_t> package(uint8_t target, uint8_t cmd, const uint8_t* data,
                               uint16_t len, uint16_t mask = 0x80);
  void parse_callback();
  bool write_to_device(uint8_t device_id);
  std::vector<uint8_t> send_sdo(uint16_t slave_id, const uint8_t* data,
                                uint16_t len, uint32_t timeout_ms = 100);
  void init_map();
  DeviceInfo_t scan_hand_info(uint8_t slave_id, uint8_t device_id = 0xFF);
  bool read_register(const uint8_t device_id, uint16_t index, uint16_t len);
  bool write_register(const uint8_t device_id, uint16_t index, uint8_t* data,
                      uint16_t len);
  void stop_update_thread();

 private:
  std::mutex mutex_;
  std::shared_ptr<std::thread> ethercat_update_thread_;
  DeviceInfo_t device_info_;
  // key is slave ids
  std::map<uint8_t, HandState_t> hand_state_;
  // key is slave ids
  std::map<uint8_t, HandCommand_t> hand_command_;
  std::atomic<bool> exit_{false};
  std::string ifname_;
  char IOmap[4096];
  std::atomic<bool> sdo_update_{false};
  // PdModePdoOutputType* pdo_output_;
  // PdModePdoInputType* pdo_input_;
  uint16_t expected_wkc_{0};
  int finger_index_{0};
  Parse_Struct_t parse_usb_cdc_;
  std::string joint_version_;
  std::map<uint8_t, uint8_t> device_id_to_slave_map_;
  uint8_t hand_register_tab_[512]{0};
};
}  // namespace ethercat_communication

#endif  // _ETHERCAT_COMMUNICATION_HPP_