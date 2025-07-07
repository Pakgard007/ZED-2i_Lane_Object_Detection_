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

# ----------------- คำนวณ Homography Matrix -----------------
src_pts = np.array([
    [300, 700], [1000, 700], [550, 300], [750, 300]   
], dtype=np.float32)

dst_pts = np.array([
    [300, 850], [900, 850], [300, 400], [900, 400]   
], dtype=np.float32)

H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

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

# ตัดสินใจเคลื่อนที่ของรถ 🚗
def decision_making(object_depths):
    if not object_depths:
        return "✅ SAFE: Continue Moving"

    min_depth = min([d for d in object_depths if d is not None], default=None)

    if min_depth is None:
        return "✅ SAFE: Continue Moving"
    elif min_depth < 3.0:
        return "🚫 STOP: Obstacle Ahead!"
    elif min_depth < 5.0:
        return "⚠ SLOW DOWN: Obstacle Ahead"
    else:
        return "✅ SAFE: Continue Moving"

# ----------------- ฟังก์ชัน Overlay Drivable Area & Lane -----------------
def overlay_segmentation(image, mask, color=(0, 255, 0), alpha=0.4):
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask > 0] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # ปรับความสว่างของภาพให้อยู่ในช่วงปกติ
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX)
    
    return blended

# ----------------- แสดง ROI เฉพาะใน Front View -----------------
def show_roi_in_front_view(image, roi, roi_coords):
    x_start, y_start, roi_width, roi_height = roi_coords
    # วาดกรอบใน Front View
    cv2.rectangle(image, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (255, 255, 255), 2)
    return image

# ----------------- แปลง Front View เป็น BEV -----------------
def transform_to_bev(image, H):
    bev = cv2.warpPerspective(image, H, (1280, 720))
    return bev

def pixel_to_real_distance(pixel_x, pixel_y, K, R, T):
    """
    แปลงพิกเซลเป็นระยะทางจริงโดยใช้ค่าคาลิเบรตของกล้อง
    """
    # สร้างพิกัดโฮโมจีเนียส (Homogeneous Coordinates)
    pixel_coords = np.array([[pixel_x], [pixel_y], [1]], dtype=np.float32)

    # แปลงเป็นพิกัดกล้อง
    camera_coords = np.linalg.inv(K) @ pixel_coords

    # แปลงจากกล้อง → พิกัดโลก
    world_coords = np.linalg.inv(R) @ (camera_coords - T)

    # คำนวณระยะทางจริง
    real_distance = np.sqrt(world_coords[0]**2 + world_coords[1]**2)

    return float(real_distance)  # แปลงเป็น float ก่อนส่งออก

def get_center_of_drivable_area_bev(da_seg_bev):
    """
    คำนวณตำแหน่งศูนย์กลางของ Drivable Area ในมุมมอง BEV
    """
    height, width = da_seg_bev.shape
    y, x = np.where(da_seg_bev > 0)  # หาพิกเซลที่เป็นพื้นที่ขับขี่

    if len(x) == 0:  # ถ้าไม่มีข้อมูล
        return width // 2  # คืนค่ากึ่งกลางของจอ

    return int(np.mean(x))  # คำนวณค่ากลางของ Drivable Area ใน BEV
    
# ฟังก์ชันตรวจว่า YOLOP เพี้ยนหรือไม่
# เช่น กรณี lane detection หาย หรือขับเบี่ยงจากศูนย์กลางมาก
    
def draw_drivable_rows(bev_image, da_seg_bev_mask, num_lines=9, slope_threshold=0.05):
    """
    - วาดเส้นแนวนอนที่อยู่ในบริเวณ Drivable Area
    - ประมาณเส้นตรงจากจุดศูนย์กลาง
    - คืนค่า True/False ว่าควรขับตรงหรือไม่
    """
    height, width = da_seg_bev_mask.shape
    y_indices = np.linspace(0, height - 1, num_lines).astype(int)
    centers = []

    for y in y_indices:
        row = da_seg_bev_mask[y]
        x_indices = np.where(row > 0)[0]  # พิกัด x ที่เป็น Drivable Area

        if len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            x_center = (x_min + x_max) // 2
            centers.append((x_center, y))
            cv2.line(bev_image, (x_min, y), (x_max, y), (0, 0, 255), 2)
            cv2.circle(bev_image, (x_center, y), 5, (0, 255, 255), -1)

    # ต้องมีจุดอย่างน้อย 2 จุดเพื่อประมาณเส้นตรง
    if len(centers) >= 4:
        pts = np.array(centers)
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)  # fit x = m*y + c
        slope = coeffs[0]

        # วาดเส้นประมาณ
        y0, y1 = 0, height - 1
        x0 = int(coeffs[0] * y0 + coeffs[1])
        x1 = int(coeffs[0] * y1 + coeffs[1])
        cv2.line(bev_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # ตัดสินใจว่าควรขับตรงหรือไม่
        if abs(slope) < slope_threshold:
            cv2.putText(bev_image, "Move Straight", (50, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return True
        else:
            cv2.putText(bev_image, "Not Straight Enough", (50, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False
    else:
        cv2.putText(bev_image, "Not Enough Points", (50, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return False

# ค่าของ Wheelbase และ Lookahead Distance
wheelbase = 1.8  # ความยาวของฐานล้อ (เมตร)
lookahead_distance = 2.0  # ระยะห่างของ Lookahead (เมตร)

# ค่า Wheelbase ควรตั้งตามข้อมูลของรถที่ใช้ (เช่น รถขนาดเล็กหรือใหญ่)
# ค่า Lookahead Distance ขึ้นอยู่กับความเร็วของรถและลักษณะของเส้นทาง

def get_lookahead_point_from_bev(car_position, da_seg_bev_mask, lookahead_distance):
    """
    ฟังก์ชันนี้ใช้ในการค้นหา Lookahead Point ใน BEV ที่อยู่ในพื้นที่ Drivable Area
    car_position: ตำแหน่งของรถ (x, y)
    da_seg_bev_mask: Mask ของ Drivable Area ใน BEV
    lookahead_distance: ระยะห่างที่ต้องการให้รถไปถึง (Lookahead Distance)
    """
    height, width = da_seg_bev_mask.shape
    for y in range(height):
        for x in range(width):
            if da_seg_bev_mask[y, x] > 0:  # หากพิกเซลอยู่ใน Drivable Area
                distance = np.linalg.norm(np.array([x, y]) - car_position)
                if distance >= lookahead_distance:
                    return x, y  # คืนค่า Lookahead point
    return width - 1, height - 1  # หากไม่พบ, ใช้จุดสุดท้าย

def calculate_steering_angle(car_position, lookahead_point, wheelbase):
    """
    คำนวณมุมการเลี้ยวจาก Lookahead Point ตามสูตร Pure Pursuit
    car_position: ตำแหน่งของรถ (x, y)
    lookahead_point: จุดที่รถต้องไป (x, y)
    wheelbase: ความยาวของฐานล้อ
    """
    dx = lookahead_point[0] - car_position[0]
    dy = lookahead_point[1] - car_position[1]
    angle_to_target = np.arctan2(dy, dx)  # มุมจากรถไปยัง Lookahead Point

    # คำนวณมุมการเลี้ยว (ตามสูตร Pure Pursuit)
    steering_angle = np.arctan2(2 * wheelbase * np.sin(angle_to_target), lookahead_distance)
    return steering_angle

def get_lookahead_point_from_da_mask(car_position, da_seg_mask, lookahead_distance):
    """
    ฟังก์ชันนี้ใช้ในการค้นหา Lookahead Point จาก Mask ของ Drivable Area ใน Front View
    car_position: ตำแหน่งของรถ (x, y)
    da_seg_mask: Mask ของ Drivable Area
    lookahead_distance: ระยะห่างที่ต้องการให้รถไปถึง (Lookahead Distance)
    """
    height, width = da_seg_mask.shape
    lookahead_point = None

    # ค้นหาพื้นที่ขับขี่ที่ใกล้ที่สุดในขอบเขต lookahead
    for y in range(height):
        for x in range(width):
            if da_seg_mask[y, x] > 0:  # พิกเซลใน Drivable Area
                distance = np.linalg.norm(np.array([x, y]) - car_position)

                if distance >= lookahead_distance:
                    lookahead_point = (x, y)
                    break

        if lookahead_point is not None:
            break

    # ถ้าไม่พบ Lookahead Point ในพื้นที่ Drivable Area
    if lookahead_point is None:
        print("No valid lookahead point found!")
        return None

    return lookahead_point

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-Front-latest3.mp4', fourcc, 30.0, (1280, 720))
out1 = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-BEV-latest3.mp4', fourcc, 30.0, (1280, 720))

# ----------------- อ่านภาพจากกล้อง ZED 2i และตรวจจับ -----------------
prev_center = None
trajectory_points = []

image = sl.Mat()
depth_map = sl.Mat()

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # ตรวจจับ Drivable Area และ Lane
        da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(undistorted_frame, model)

        # ----------------- สร้าง BEV -----------------
        bev_image = transform_to_bev(undistorted_frame.copy(), H)
        
        # วาดเส้นทางย้อนหลัง (จุดสีแดง)
        for pt in trajectory_points:
            cv2.circle(bev_image, pt, 2, (0, 0, 255), -1)

        # ----------------- แสดงผล BEV -----------------
        cv2.imshow("Bird's Eye View", bev_image)  # แสดงผล BEV ที่มีการตรวจจับและข้อความ

        # ----------------- Overlay Drivable Area & Lane บน Front View -----------------
        overlay_da = overlay_segmentation(undistorted_frame.copy(), da_seg_mask, (0, 255, 0))  # ซ้อนสีเขียว (Drivable Area)
        overlay_ll = overlay_segmentation(overlay_da, ll_seg_mask, (0, 0, 255))  # ซ้อนสีแดง (Lane)

        # ----------------- สร้าง BEV โดยใช้ overlay_ll (ไม่มี ROI) -----------------
        bev_image = transform_to_bev(overlay_ll.copy(), H)  # ใช้ overlay_ll ที่ไม่มี ROI
        scale = bev_image.shape[1] / 12.8   # 100 pixels = 1 meter → 12.8 m = 1280 px
        
        # 🟡 ตรวจสอบศูนย์กลางบนแนวขวางหลายจุด
        da_seg_mask_uint8 = (da_seg_mask > 0).astype(np.uint8) * 255
        da_seg_bev_mask = cv2.warpPerspective(da_seg_mask_uint8, H, (1280, 720))

        # 🔹 สมมุติตำแหน่งของรถบนภาพ BEV (กลางล่างของภาพ)
        car_position = np.array([640, 700])

        # 🔹 หา Lookahead Point จาก BEV mask
        lookahead_point = get_lookahead_point_from_da_mask(car_position, da_seg_bev_mask, lookahead_distance)

        # 🔹 คำนวณมุมการเลี้ยวจาก Pure Pursuit
        steering_angle = calculate_steering_angle(car_position, lookahead_point, wheelbase)

        # 🔹 วาดตำแหน่งรถ (สีเขียว)
        cv2.circle(bev_image, tuple(car_position), 10, (0, 255, 0), -1)
        cv2.putText(bev_image, 'Car Position', (car_position[0] - 80, car_position[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🔹 วาด Lookahead Point (สีน้ำเงิน)
        cv2.circle(bev_image, tuple(lookahead_point), 10, (255, 0, 0), -1)
        cv2.putText(bev_image, 'Lookahead Point', (lookahead_point[0] - 100, lookahead_point[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 🔹 วาดเส้นเชื่อม (สีเหลือง)
        cv2.line(bev_image, tuple(car_position), tuple(lookahead_point), (255, 255, 0), 2)

        # 🔹 แสดงมุมเลี้ยวใน terminal
        print(f"Steering Angle: {math.degrees(steering_angle):.2f} degrees")
        
        should_go_straight = draw_drivable_rows(bev_image, da_seg_bev_mask, num_lines=9)

        if should_go_straight:
            speed = 0.4
            steering_command = 0
        else:
            speed = 0.0
            steering_command = 0
            print("❗ เส้นไม่ตรงพอ → ไม่สั่งให้เคลื่อนที่")

        # ----------------- เพิ่มข้อความใน Front View -----------------
        # ตั้งค่า default เป็น overlay_ll
        front_view_with_distance = overlay_ll.copy()
        cv2.imshow("Bird's Eye View", bev_image)  # BEV มีแค่ Drivable Area & Lane (ไม่มี ROI และ Bounding Box)

        # รอให้กดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_with_da = overlay_segmentation(frame.copy(), da_seg_mask, (0, 255, 0))  
        frame_with_ll = overlay_segmentation(frame_with_da, ll_seg_mask, (0, 0, 255))  
        roi, roi_coords = create_roi(frame_with_ll)
        object_depths = get_depth_for_detections(depth_map.get_data(), det_pred[0], roi_coords)

        # แสดงผลลัพธ์ใน Terminal
        if object_depths:
            for i, depth in enumerate(object_depths):
                if depth is not None:
                    print(f"▶ Object {i+1}: {depth:.2f} m")
                else:
                    print(f"▶ Object {i+1}: No Depth Data")

        # วาด Bounding Box และหมายเลขวัตถุ + ระยะห่าง
        if det_pred[0] is not None:
            for i, det in enumerate(det_pred[0]):
                if len(det) < 6:
                    continue  

                x1, y1, x2, y2, conf, cls = det[:6]

                # ✅ เงื่อนไขกรอบต้องอยู่ใน ROI
                x_roi, y_roi, roi_width, roi_height = roi_coords
                if x1 < x_roi or x2 > (x_roi + roi_width) or y1 < y_roi or y2 > (y_roi + roi_height):
                    continue  # ข้ามวัตถุที่อยู่นอก ROI

                obj_id = i + 1
                label = f'Object {obj_id}'

                if object_depths[i] is not None:
                    depth_text = f'{object_depths[i]:.2f} m'
                    label += f' ({depth_text})'
                else:
                    depth_text = "No Depth Data"

                plot_one_box((x1, y1, x2, y2), frame_with_ll, label=label, color=(0, 0, 255), line_thickness=2)
                cv2.putText(frame_with_ll, str(obj_id), (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                print(f"▶ Object {obj_id}: {depth_text}")


        decision = decision_making(object_depths)
        print(f"🛑 Driving Decision: {decision}")
        cv2.imshow("Front View - YOLOP", frame_with_ll)
    
        # ----------------- แสดงผล Front View -----------------
        out.write(frame_with_ll)
        out1.write(bev_image)

        center_da = get_center_of_drivable_area_bev(da_seg_mask)


# ----------------- ปิดกล้อง -----------------
out.release()
out1.release()
zed.close()
cv2.destroyAllWindows()

