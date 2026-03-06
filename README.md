# Unitree-Go2-GestureDetection

🤖 Control your Unitree Go2 robot with hand gestures!

[English](#english) | [中文](#中文)

---

## English

### 🎯 Overview

This project enables gesture-based control of the Unitree Go2 quadruped robot using MediaPipe hand tracking and the Unitree SDK2 Python API.

### ✨ Features

- **🤚 Hand Gesture Recognition** - Detect wrist flips and hand poses
- **🐕 Robot Control** - Execute commands without remote controller
- **📹 Real-time Video Stream** - Process robot's front camera feed
- **🎮 Multiple Gestures**:
  - Wrist flip → Stand / Sit
  - ✌️ Gesture 1 (Index + Middle finger up) → Hello
  - 🤟 Gesture 2 (Victory sign) → Heart

### 🛠️ Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 📦 Dependencies

See `requirements.txt` for full list:

- `mediapipe` - Google's hand tracking solution
- `opencv-python` - Computer vision
- `numpy` - Numerical computing
- `unitree-sdk2-python` - Unitree robot SDK

### 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lez666/Unitree-Go2-GestureDetection.git
   cd Unitree-Go2-GestureDetection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Unitree SDK**
   ```bash
   # Follow official guide: https://github.com/unitreerobotics/unitree_sdk2_python
   ```

### 🎯 Usage

#### Option 1: With Visual Interface (Recommended)

Run on your laptop with display:

```bash
python3 hand_dog_cv.py <network_interface>
```

Example:
```bash
python3 hand_dog_cv.py enp7s0
```

#### Option 2: Headless Mode

SSH into the robot's Jetson extension board:

```bash
python3 hand_dog_nonscreen.py <network_interface>
```

Example:
```bash
python3 hand_dog_nonscreen.py eth0
```

### 📋 Network Interface

Find your network interface:

```bash
# Linux
ip addr

# macOS
ifconfig
```

Common interfaces:
- `enp7s0` - Ethernet (Linux)
- `eth0` - Ethernet (Jetson)
- `wlan0` - WiFi (Linux)

### 🎓 How It Works

1. **Video Capture** - Get frames from Go2's front camera
2. **Hand Detection** - MediaPipe processes the video stream
3. **Gesture Recognition** - Analyze hand landmarks:
   - Wrist position determines flip direction
   - Finger positions identify specific gestures
4. **Robot Control** - Send commands via Unitree SDK

### ⚠️ Troubleshooting

**Robot not responding?**
- Check network connectivity
- Verify SDK initialization
- Ensure robot is in sport mode

**Gesture not detected?**
- Ensure good lighting
- Keep hand within camera frame
- Make distinct gestures

### 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

### 📄 License

MIT License

### 👤 Author

**Enze Li**
- GitHub: [@lez666](https://github.com/lez666)
- Email: support_lez@unitree.com

---

## 中文

### 🎯 项目简介

基于 MediaPipe 手势识别和 Unitree SDK2 Python API，实现用手势控制宇树 Go2 机器狗。

### ✨ 功能特性

- **🤚 手势识别** - 检测手腕翻转和手势
- **🐕 机器人控制** - 无需遥控器执行命令
- **📹 实时视频流** - 处理机器狗前置摄像头
- **🎮 支持手势**:
  - 手腕翻转 → 站起 / 坐下
  - ✌️ 手势 1 (食指+中指向上) → Hello
  - 🤟 手势 2 (胜利手势) → 比心

### 🛠️ 环境要求

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 📦 依赖列表

详见 `requirements.txt`:
- `mediapipe` - Google 手势追踪
- `opencv-python` - 计算机视觉
- `numpy` - 数值计算
- `unitree-sdk2-python` - 宇树机器人 SDK

### 🚀 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/lez666/Unitree-Go2-GestureDetection.git
   cd Unitree-Go2-GestureDetection
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装宇树 SDK**
   ```bash
   # 参考官方指南: https://github.com/unitreerobotics/unitree_sdk2_python
   ```

### 🎯 使用方法

#### 方式一：带可视化界面 (推荐)

在笔记本上运行：

```bash
python3 hand_dog_cv.py <网络接口>
```

示例:
```bash
python3 hand_dog_cv.py enp7s0
```

#### 方式二：无界面模式

SSH 进入机器狗的 Jetson 扩展板：

```bash
python3 hand_dog_nonscreen.py <网络接口>
```

示例:
```bash
python3 hand_dog_nonscreen.py eth0
```

### 📋 网络接口

查找网络接口:

```bash
# Linux
ip addr

# macOS
ifconfig
```

常用接口:
- `enp7s0` - 以太网 (Linux)
- `eth0` - 以太网 (Jetson)
- `wlan0` - WiFi (Linux)

### 🎓 工作原理

1. **视频采集** - 获取 Go2 前置摄像头画面
2. **手势检测** - MediaPipe 处理视频流
3. **手势识别** - 分析手部关键点:
   - 手腕位置判断翻转方向
   - 手指位置识别特定手势
4. **机器人控制** - 通过 Unitree SDK 发送指令

### ⚠️ 故障排除

**机器人无响应?**
- 检查网络连接
- 验证 SDK 初始化
- 确保机器狗处于运动模式

**手势检测不到?**
- 确保光线充足
- 手保持在画面内
- 做明显的手势

### 🤝 贡献

欢迎提交 Issue 和 PR！

### 📄 许可证

MIT License

### 👤 作者

**李恩泽 (Enze Li)**
- GitHub: [@lez666](https://github.com/lez666)
- Email: support_lez@unittree.com

---

⭐ Star us if this helps!
