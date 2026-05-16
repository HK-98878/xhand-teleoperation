# XHandControlROS 使用说明

## 介绍
XHandControlROS 是一个 ROS 包，用于与 XHand 机械手进行 EtherCAT 通信，控制机械手的关节和读取传感器数据。该包通过 ROS 话题、服务和回调函数提供对 XHand 的全面控制接口。

## 安装
确保已经安装了以下依赖：

- ROS (支持 ROS noetic)

克隆此包到你的 catkin_ws 工作空间并进行编译：

```bash
cd ~/catkin_ws/src
git clone <your-repository-url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
## 节点

### xhand_control_ros 节点
该节点负责与 XHand 设备建立连接，处理手的状态和指令。

### 启动节点
你可以用`root用户`通过运行以下命令启动节点：

```bash
source devel/setup.bash
roslaunch xhand_control_ros xhand_control.launch
```
或者，你可以用`普通用户`通过运行以下命令启动节点：
- 每次编译完代码后，给可执行文件添加权限：
```bash
sudo chown root:root devel/lib/xhand_control_ros/xhand_control_ros_node
sudo chmod u+s devel/lib/xhand_control_ros/xhand_control_ros_node
```
- 然后运行节点：
```bash
source devel/setup.bash
roslaunch xhand_control_ros xhand_control.launch
```
## 话题

### 发布的话题

- `/xhand_control/xhand_state`：发布 XHand 的关节状态及传感器数据。
  - 消息类型：`xhand_control_ros::XHandStateArray`
  - XHandStateArray 消息内容：
    - `header`：消息头。
    - `hand_id[]`：手 ID。
    - `hand_name[]`：手名称。
    - `hand_states[]`：每个手的关节状态。
      - `name[]`：关节名称。
      - `position[]`：关节位置。
      - `effort[]`：关节力矩。
      - `temperature[]`：关节温度。
      - `error_code[]`：关节错误代码。
    - `sensor_states[]`：每个手的传感器状态。
      - `finger_sensor_states[]`：单个手的传感器状态。
        - `location`：传感器位置。
        - `calc_force[]`：传感器合力
        - `raw_force[]`：传感器原始力数据。
        - `calc_temperature[]`：传感器扭矩。
        - `raw_temperature[]`：传感器原始扭矩数据。
### 查看消息示例

```bash
rostopic echo /xhand_control/xhand_state
```

### 订阅的话题

- `/xhand_control/xhand_command`：订阅用户发送的 XHand 控制命令。
  - 消息类型：`xhand_control_ros::XHandCommand`
  - XHandCommand 消息内容：
    - `hand_id`：指定要控制的手ID。
    - `name[]`：指定要控制的关节名称列表。
    - `position[]`：各个关节目标位置。
    - `kp[]`, `kd[]`, `ki[]`：PID 控制参数。
    - `effort_limit[]`：力矩限制。
    - `mode`：控制模式。

### 发布消息示例

```bash
rostopic pub -1 /xhand_control/xhand_command xhand_control_ros/XHandCommand '{hand_id: 0, name: ["thumb_bend_joint", "index_bend_joint"], position: [0.5, 0.18], kp: [100, 100], kd: [0, 0], ki: [0, 0], effort_limit: [350, 350], mode: 3}'
```

## 服务

### 提供的服务

- `/xhand_control/read_hand_info`：读取 XHand 的信息。
  - 服务类型：`xhand_control_ros::ReadHandInfo`
  - 请求字段：
    - `hand_id`：要查询的手的 ID。
    - `info_type`：查询类型 (如手的类型、序列号、版本等)。
  - 响应字段：
    - `success`：操作是否成功。
    - `info`：返回的手信息。

- `/xhand_control/set_hand_id`：设置 XHand 的 ID。
  - 服务类型：`xhand_control_ros::SetHandId`
  - 请求字段：
    - `current_id`：当前的手 ID。
    - `new_id`：要设置的新 ID。
  - 响应字段：
    - `success`：操作是否成功,成功后需要重新启动节点。

- `/xhand_control/set_hand_name`：设置 XHand 的名称。
  - 服务类型：`xhand_control_ros::SetHandName`
  - 请求字段：
    - `hand_id`：要设置的手的 ID。
    - `hand_name`：要设置的新名称。
  - 响应字段：
    - `success`：操作是否成功,成功后需要重新启动节点。

- `/xhand_control/reset_hand_sensor`：重置 XHand 的传感器。
  - 服务类型：`xhand_control_ros::ResetSensor`
  - 请求字段：
    - `hand_id`：要重置的手的 ID。
    - `sensor_id`：要重置的传感器 ID (17 到 21)。
  - 响应字段：
    - `success`：操作是否成功。

### 服务调用示例

```bash
rosservice call /xhand_control/read_hand_info '{hand_id: 0, info_type: 0}'
rosservice call /xhand_control/set_hand_name '{hand_id: 0, hand_name: 'test_hand'}'
rosservice call /xhand_control/set_hand_id '{current_id: 1, new_id: 2}'
rosservice call /xhand_control/reset_hand_sensor '{hand_id: 0, sensor_id: 20}'
```


## 常见问题
1. **无法检测到设备**  
   请确认 EtherCAT 设备连接正常，并且正确配置了接口名。

2. **无法发布或订阅消息**  
   确保你已正确启动节点，并且话题命名与实际订阅或发布的一致。
