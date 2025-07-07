import argparse
import os, sys
import time
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import pyzed.sl as sl
import torch.nn.functional as F
import pandas as pd
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# Import YOLOP utilities
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.core.general import scale_coords

# Normalize for YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ----------------- เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  
init_params.camera_fps = 60
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # ใช้ Depth Mode
init_params.coordinate_units = sl.UNIT.METER  # หน่วยเป็นเมตร  

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image = sl.Mat()
depth_map = sl.Mat()

# ----------------- โหลดค่าคาลิเบรตจากไฟล์ -----------------
calib_data = np.load("camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = np.array([
    -0.07709319689152208,  # k1
     0.06756180189133752,  # k2
     0.00015006759935512075,  # p1 (tangential distortion)
    -6.006342505065124e-05,  # p2 (tangential distortion)
    -0.028545020615709165  # k3
])

extrinsic_data = np.load("extrinsic_parameters.npz")
R = extrinsic_data["R_left"]
T = extrinsic_data["T_left"]

# ----------------- โหลด Waypoint -----------------
df = pd.read_csv("/home/mag/waypoint/Merged_Position_and_Angular_Data1.csv")
waypoints = df.to_dict(orient="records")
current_wp_index = 0

# ----------------- ฟังก์ชันช่วย -----------------
def find_next_waypoint(current_x, current_y, current_wp_index):
    """
    เปรียบเทียบตำแหน่งปัจจุบันกับ waypoint และเลือก waypoint ถัดไป
    """
    for i in range(current_wp_index, len(waypoints)):
        wp = waypoints[i]
        waypoint_x = wp['x']
        waypoint_y = wp['y']
        distance = math.sqrt((waypoint_x - current_x) ** 2 + (waypoint_y - current_y) ** 2)
        if distance > 0.5:  # ถ้าระยะห่างจาก waypoint มากกว่า 0.5 เมตร ให้เลือก waypoint ถัดไป
            return wp, i
    return waypoints[-1], current_wp_index  # ถ้าไม่มี waypoint ถัดไปในระยะที่กำหนด ให้คืนค่า waypoint สุดท้าย

# ----------------- Subscriber สำหรับรับข้อมูลจาก topic positionxyz -----------------
class PositionXYZListener(Node):
    def __init__(self):
        super().__init__('positionxyz_listener')

        # Subscriber สำหรับรับข้อมูลจาก topic positionxyz
        self.create_subscription(
            PoseStamped,
            '/positionxyz',  # Topic ที่ส่งค่าตำแหน่ง (x, y) และมุม angular_z
            self.position_callback,
            10
        )

    def position_callback(self, msg):
        current_x = msg.pose.position.x
        current_y = msg.pose.position.y
        current_angular_z = msg.pose.orientation.z  # หรือถ้าค่า angular_z อยู่ในทิศทางอื่นก็เปลี่ยน

        print(f"ตำแหน่งปัจจุบัน: x={current_x}, y={current_y}, angular_z={current_angular_z}")

        # ค้นหา waypoint ถัดไป
        next_wp, next_wp_index = self.find_next_waypoint(current_x, current_y)

        print(f"Waypoint ถัดไป: {next_wp['waypoint']} ที่ตำแหน่ง x={next_wp['x']}, y={next_wp['y']}")

        # อัปเดต waypoint index
        global current_wp_index
        current_wp_index = next_wp_index

    def find_next_waypoint(self, current_x, current_y):
        """
        เปรียบเทียบตำแหน่งปัจจุบันกับ waypoint และเลือก waypoint ถัดไป
        """
        for i in range(current_wp_index, len(waypoints)):
            wp = waypoints[i]
            waypoint_x = wp['x']
            waypoint_y = wp['y']
            distance = math.sqrt((waypoint_x - current_x) ** 2 + (waypoint_y - current_y) ** 2)
            if distance > 0.5:  # ถ้าระยะห่างจาก waypoint มากกว่า 0.5 เมตร ให้เลือก waypoint ถัดไป
                return wp, i
        return waypoints[-1], current_wp_index  # ถ้าไม่มี waypoint ถัดไปในระยะที่กำหนด ให้คืนค่า waypoint สุดท้าย

#-------------------Create ROI----------------------
def create_roi(image, roi_width_pixels=150, roi_height_pixels=500):

    height, width = image.shape[:2]
    x_start = (width - roi_width_pixels) // 2
    y_start = (height - roi_height_pixels) // 2

    # วาด ROI ลงบนภาพที่ส่งเข้ามาโดยตรง
    cv2.rectangle(image, (x_start, y_start), (x_start + roi_width_pixels, y_start + roi_height_pixels), (255, 255, 255), 2)

    roi = image[y_start:y_start + roi_height_pixels, x_start:x_start + roi_width_pixels]
    
    return roi, (x_start, y_start, roi_width_pixels, roi_height_pixels)

# ----------------- โหลดโมเดล YOLOP -----------------
def load_yolop_model(weights_path, device):
    model = get_net(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model Loaded: {weights_path}")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")
model = load_yolop_model("/home/mag/satoi/python/YOLOP/weights/End-to-end.pth", device)

# ----------------- ฟังก์ชันดึง Depth ของวัตถุที่ตรวจจับได้ -----------------
def get_depth_for_detections(depth_map, detections, roi_coords):
    object_depths = []
    
    if detections is None or len(detections) == 0:
        return object_depths  

    x_roi, y_roi, roi_width, roi_height = roi_coords

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  

        # เช็ควัตถุอยู่ใน ROI หรือไม่
        if x1 < x_roi or x2 > (x_roi + roi_width) or y1 < y_roi or y2 > (y_roi + roi_height):
            object_depths.append(None)
            continue  

        obj_depth_roi = depth_map[y1:y2, x1:x2]
        valid_depth = obj_depth_roi[np.isfinite(obj_depth_roi)]

        if len(valid_depth) == 0:
            object_depths.append(None)
        else:
            object_depths.append(np.median(valid_depth))
    
    return object_depths

# ----------------- ตรวจจับ Object, Lane และ Drivable Area -----------------
def detect_obstacles(image, model):
    input_image = transform(image).to(device)
    input_image = input_image.unsqueeze(0)

    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(input_image)

    _, _, height, width = input_image.shape
    da_seg_out = F.interpolate(da_seg_out, size=(height, width), mode='bilinear', align_corners=False)
    ll_seg_out = F.interpolate(ll_seg_out, size=(height, width), mode='bilinear', align_corners=False)

    da_seg_mask = torch.max(da_seg_out, 1)[1].squeeze().cpu().numpy()
    ll_seg_mask = torch.max(ll_seg_out, 1)[1].squeeze().cpu().numpy()
    det_pred = non_max_suppression(det_out[0], conf_thres=0.3, iou_thres=0.5)

    # ตรวจสอบผลลัพธ์จาก non_max_suppression
    return da_seg_mask, ll_seg_mask, det_pred

# ----------------- ฟังก์ชันหลัก -----------------
def main(args=None):
    rclpy.init(args=args)

    # เริ่มต้น ROS2 node
    position_listener = PositionXYZListener()

    # รัน ROS2 Node
    rclpy.spin(position_listener)

    # ปิด ROS2 เมื่อเสร็จสิ้น
    position_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
