#ifndef _COMMUNIATION_INTERFACE_H_
#define _COMMUNIATION_INTERFACE_H_
#include <map>
#include <string>
#include <vector>

#include "data_type.hpp"
#include "ev_boot.h"

// #define APP_ADDR_OFFSET 512  // APP运行区实际代码偏移量
// #define BOOT_DATA_BUF_INDEX_LEN 2
// #define BOOT_DATA_BUF_DATA_LEN 4096
// #define BOOT_DATA_BUF_LEN (BOOT_DATA_BUF_INDEX_LEN + BOOT_DATA_BUF_DATA_LEN)
// //升级指令
// #define CMD_IAP_BEGIN 0xA1   //升级开始
// #define CMD_IAP_TRANS 0xA2   //升级数据
// #define CMD_IAP_VERIFY 0xA3  //升级校验
// #define CMD_IAP_RESET 0xA4   // MCU复位
// #define CMD_IAP_ACK 0xB0     //升级应答
// //升级错误码
// #define IAP_OK 0
// #define IAPERROR_SIZE 1        //固件大小超范围
// #define IAPERROR_ERASE 2       //擦除Flash失败
// #define IAPERROR_WRITEFLASH 3  //写入Flash失败
// #define IAPERROR_INDEX 5       //索引错误
// #define IAPERROR_BUSY 6        //总线忙
// #define IAPERROR_FORM 7        //数据格式错误
// #define IAPERROR_VERIFY 8      //校验错误
// #define IAPERROR_OTHER 9       //其他错误

#define CMD_SET_EXPECT_PARAM 0x02       //设置指关节期望参数
#define CMD_READ_USER_PARAM 0x03        //读取指关节用户参数
#define CMD_WRITE_USER_PARAM 0x04       //写入指关节用户参数
#define CMD_BURN_FLASH 0x05             //指关节写参数到flash中
#define CMD_UPDATE_CURRENT_PARAM 0x06   //更新电流环参数
#define CMD_UPDATE_POSITION_PARAM 0x07  //更新位置环参数
#define CMD_SET_MODE 0x08               //设置模式
#define CMD_READ_MODE 0x09              //读取模式
#define CMD_QUERY_MATRIX_PRESSURE 0x11  //获取传感器数据
#define CMD_CLREAR_PRESSURE 0x12        //传感器数据清零
#define CMD_READ_SOFTWARE_VERSION 0X13  // 读取软件版本号
#define CMD_RESET 0X14                  // 软件复位重启
#define CMD_READ_REGISTER 0X15          // 读取寄存器数据
#define CMD_WRITE_REGISTER 0X16         // 写入寄存器数据

#define CMD_ERROR 0x00  // error case

//寄存器表
#define REGISTER_TAB_SIZE 512

//寄存器地址 以下只读
#define REGISTER_WHICH_HAND 20  //哪一只手 1BYTE
#define REGISTER_SN 21          // SN码 32BYTE

//寄存器地址 以下可读可写
#define REGISTER_HAND_ID 53      // ID 1byte
#define REGISTER_HAND_NAME 54    //手名字  32字节
#define REGISTER_CALIBRATION 86  //校准参数 120字节
namespace communication_interface {
class CommunicationInterface {
 public:
  virtual std::vector<std::string> enumerate_devices() {
    // C++代码实现
    return {};
  }
  virtual bool open_device() = 0;
  virtual std::vector<uint8_t> list_hands_id() = 0;
  virtual DeviceInfo_t read_device_info(uint8_t device_id) = 0;

  virtual bool send_command(const uint8_t device_id,
                            const HandCommand_t& command) = 0;
  virtual HandState_t read_state(uint8_t device_id) = 0;

  virtual HandParam_t read_joint_parameters(uint8_t device_id) = 0;

  virtual bool set_joint_parameters(const uint8_t device_id,
                                    const HandParam_t& parameters) = 0;

  virtual bool upgrade_device(const uint8_t device_id,
                              const uint8_t component_id,
                              const std::string& firmware_path) {
    // C++代码实现
    return true;
  }

  // virtual bool calibrate_joint(uint8_t device_id) {
  //   // C++代码实现
  //   return true;
  // }

  virtual std::string get_firmware_version(const uint8_t device_id,
                                           const uint8_t component_id) = 0;

  virtual bool reset_sensor(const uint8_t device_id,
                            const uint8_t sensor_id) = 0;

  virtual void close_device() {}

  virtual char get_hand_type(uint8_t device_id) = 0;
  virtual std::string get_serial_number(uint8_t device_id) = 0;
  virtual bool set_hand_name(const uint8_t device_id,
                             const std::string& name) = 0;
  virtual std::string get_hand_name(uint8_t device_id) = 0;
  virtual bool set_hand_id(uint8_t old_id, uint8_t new_id) = 0;
  // virtual uint8_t get_hand_id(uint8_t device_id) = 0;
  virtual bool set_hand_parameters(const uint8_t device_id,
                                   HandParam_t& parameters) = 0;
  virtual HandParam_t get_hand_parameters(uint8_t device_id) = 0;

 protected:
  std::shared_ptr<ev_boot::EvBoot> ev_boot_;

 private:
};
}  // namespace communication_interface

#endif