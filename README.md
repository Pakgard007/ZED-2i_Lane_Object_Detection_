
# 🚗 Autonomous Driving System with YOLOP + ZED 2i + ROS 2

This repository contains code for a complete perception and control system for an autonomous vehicle using:

- **YOLOP** for Drivable Area and Lane Detection  
- **ZED 2i Stereo Camera** for 3D object detection and depth estimation  
- **ROS 2** for real-time communication and vehicle control  
- Optional camera calibration tools and fine-tuning script for YOLOP

---

## 📁 Project Structure

```
.
├── calibration/                # Calibrate ZED 2i stereo camera
├── config/                     # YOLOP training configuration
├── interface/                  # Environment perception and object tracking
├── drive_control/             # ROS 2 Node to control vehicle movement
├── weights/                   # Pretrained YOLOP model (e.g. End-to-end.pth)
└── README.md                  # You're here!
```

---

## 🔧 1. Camera Calibration (ZED 2i)

Use this script to calibrate your stereo ZED camera and estimate extrinsic parameters.

📄 `calibration/zed2i_calibration.py`

- Finds checkerboard in stereo images  
- Calculates and saves Rotation and Translation matrices  
- Output: `extrinsic_parameters.npz`

---

## 🧠 2. YOLOP Fine-Tuning (Optional)

Modify and train YOLOP with custom datasets using the provided config structure.

📄 `config/yolop_config.py`

- Dataset paths  
- Hyperparameters  
- Task selection: Detection / Lane / Drivable Area  
- Supports single-task or end-to-end training

> ⚠️ You must set paths to your own datasets and weights before training.

---

## 👁️‍🗨️ 3. Front Environment Perception

The main interface combines ZED SDK + YOLOP model to detect:

- Drivable area (green)  
- Lane lines (red)  
- People and vehicles in 3D with distance  
- Decision making: `"go"`, `"slow"`, `"brake"`

📄 `interface/main_interface.py`

- Publishes `/drive_status` topic (`String`): `"go"`, `"slow"`, or `"brake"`  
- Visual output with lane overlay and object tracking

---

## 🕹️ 4. Drive Control Node (ROS 2)

Receives `/cmd_speed`, `/cmd_angle`, and `/drive_status`, then publishes `/final_cmd_vel` as `geometry_msgs/Twist`.

📄 `drive_control/drive_control_node.py`

Behavior:
- Full speed on `"go"`  
- Gradual slowdown on `"slow"`  
- Immediate stop on `"brake"`

---

## ✅ Requirements

- Python ≥ 3.8  
- ROS 2 (Humble recommended)  
- PyTorch  
- ZED SDK (with `pyzed.sl`)  
- OpenCV  
- YACS

---

## ▶️ Running the System

Start the perception system:

```bash
python3 interface/main_interface.py
```

Start the ROS 2 control node:

```bash
ros2 run drive_control drive_control_node
```

> Ensure ZED camera is connected and ROS 2 is sourced before running.

---

## 📸 Example Output

### Drivable Area and Lane Detection
![Figure 1](https://raw.githubusercontent.com/Pakgard007/ZED-2i_Lane_Object_Detection_/main/image/12.png)

### Vehicle Detection
![Figure 2](https://raw.githubusercontent.com/Pakgard007/ZED-2i_Lane_Object_Detection_/main/image/13.png)

### Pedestrian and Vehicle Detection
![Figure 3](https://raw.githubusercontent.com/Pakgard007/ZED-2i_Lane_Object_Detection_/main/image/14.png)

### Drive Status Output
![Figure 4](https://raw.githubusercontent.com/Pakgard007/ZED-2i_Lane_Object_Detection_/main/image/15.png)

### Final Velocity Command
![Figure 5](https://raw.githubusercontent.com/Pakgard007/ZED-2i_Lane_Object_Detection_/main/image/16.png)

---

## 📬 Contact

For questions or contributions, feel free to open an [Issue](https://github.com/Pakgard007/ZED-2i_Lane_Object_Detection_/issues) or [Pull Request](https://github.com/Pakgard007/ZED-2i_Lane_Object_Detection_/pulls).
