#ifndef _DATA_TYPE_HPP_
#define _DATA_TYPE_HPP_

#include <array>
#include <cstdint>

typedef struct {
  std::array<int16_t, 12> position_stretching{};
  std::array<int16_t, 12> position_closing{};
  std::array<int16_t, 12> position_zero{};
  std::array<int16_t, 12> position_init{};
  std::array<int8_t, 12> angle_stretching{};
  std::array<int8_t, 12> angle_closing{};
} __attribute__((packed)) HandParam_t;

typedef struct {
  uint16_t resverse;
  char ev_hand{'N'};  //左手 ‘L’/‘l’ 还是右手 ‘R’/‘r’
  uint8_t hand_id;    //灵巧手ID
  std::array<char, 32> serial_number;  //序列号
  std::array<char, 32> name;           //灵巧手名称
  HandParam_t hand_param;              //校准参数
} __attribute__((packed)) DeviceInfo_t;

typedef struct {
  int8_t fx;
  int8_t fy;
  uint8_t fz;
} __attribute__((packed)) PXSR_ForceData;

typedef struct {
  PXSR_ForceData calc_force;
  std::array<PXSR_ForceData, 120> raw_force;
  std::array<uint8_t, 20> temperature;
  uint8_t calc_temperature;
} __attribute__((packed)) SenserData_t;

typedef struct {
  uint64_t id : 8; /* Subindex1 - cur_id  5bit*/
  uint64_t sensor_id : 8;
  int64_t position : 32;  /* Subindex2 - current_pos 12bit*/
  uint64_t torque : 16;   /* Subindex3 - cur_torq 9bit*/
  uint64_t default0 : 16; /* Subindex4 - res1 */
  uint64_t default1 : 16;
  uint64_t default2 : 16; /* Subindex5 - res2 */
  uint64_t default3 : 16; /* Subindex6 - res3 */
  uint64_t default4 : 16; /* Subindex7 - res4 */
  uint64_t default5 : 16;
  uint64_t default6 : 16;
  uint64_t default7 : 16;
} __attribute__((packed)) RawFingerState_t;

typedef struct {
  RawFingerState_t raw_finger_state;  //
  SenserData_t sensor_data;  //一个指尖传感器数据，uint64_t[48]
} __attribute__((packed)) UpState_t;  //上报数据结构体

typedef struct {
  std::array<RawFingerState_t, 12> finger_state;
  std::array<SenserData_t, 5> senser_data;
} __attribute__((packed)) SerialHandState_t;

typedef struct {
  uint64_t id;
  float position;
  uint64_t torque;
  uint64_t default0;
  uint64_t default1;
  uint64_t default2;
  uint64_t default3;
  uint64_t default4;
  uint64_t default5;
  uint64_t default6;
  uint64_t default7;
} __attribute__((packed)) FingerState_t;

typedef struct {
  std::array<FingerState_t, 12> finger_state;
  std::array<SenserData_t, 5> senser_data;
} __attribute__((packed)) HandState_t;

typedef struct {
  uint16_t id;
  int16_t kp;        // 比例增益参数
  int16_t ki;        // 积分系数
  int16_t kd;        // 微分增益参数
  float position;    // 设定位置
  int16_t tor_max;  // 力矩上限
  uint16_t mode;     // 模式
  uint16_t res0;     // force x (L8), force y (H8)
  uint16_t res1;     // force z (L8),
  uint16_t res2;
  uint16_t res3;
} __attribute__((packed)) FingerCommand_t;

typedef struct {
  std::array<FingerCommand_t, 12> finger_command;
} __attribute__((packed)) HandCommand_t;

#endif  // _DATA_TYPE_HPP_