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

# ----------------- โหลด Waypoint -----------------
df = pd.read_csv("/home/mag/waypoint/GPS_with_PositionXY_updated.csv")
waypoints = df.to_dict(orient="records")
current_wp_index = 0

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

# ----------------- รับข้อมูลเลนซ้าย -----------------
def get_left_lane_position(ll_seg_mask):
    """ หาเส้นขอบเลนซ้ายสุดในภาพ BEV """
    lane_pixels = np.where(ll_seg_mask > 0)  # หาจุดที่เป็นเลน
    if len(lane_pixels[0]) == 0:
        return None  # ถ้าไม่มีเลนซ้ายให้คืนค่า None

    # หาจุดที่ใกล้กับกึ่งกลางภาพที่สุด
    lane_x = lane_pixels[1]  # ค่า x ของจุดเลน
    lane_y = lane_pixels[0]  # ค่า y ของจุดเลน
    center_x = ll_seg_mask.shape[1] // 2  # กึ่งกลางภาพในแนว x

    # หาจุดเลนซ้ายที่ใกล้ center_x ที่สุด
    left_lane_x = np.min(lane_x)  # ค่าต่ำสุดของ x คือขอบเลนซ้าย
    left_lane_y = lane_y[np.argmin(lane_x)]  # y ที่ตรงกับค่า x ซ้ายสุด
    return (left_lane_x, left_lane_y)


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
    
# ----------------- ฟังก์ชันช่วย -----------------
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def find_lookahead_point(car_x, car_y, waypoints, lookahead_distance):
    for wp in waypoints:
        dx = wp["Position_X"] - car_x
        dy = wp["Position_Y"] - car_y
        dist = math.sqrt(dx**2 + dy**2)
        if dist >= lookahead_distance:
            return wp
    return waypoints[-1] # กรณีไม่มีจุดที่ห่างพอ

def pure_pursuit_control(car_x, car_y, car_theta, wp, L=1.2, L_d=2.0):
    dx = wp["Position_X"] - car_x
    dy = wp["Position_Y"] - car_y
    target_theta = math.atan2(dy, dx)
    alpha = normalize_angle(target_theta - car_theta)
    delta = math.atan2(2 * L * math.sin(alpha), L_d)

    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance > 2.0:
        speed = 0.5
    elif distance > 0.5:
        speed = 0.3
    else:
        speed = 0.0

    return delta, speed, distance

# ----------------- เริ่มระบบ -----------------
mode = "yolop"
car_x = waypoints[0]["Position_X"]
car_y = waypoints[0]["Position_Y"]
car_theta = waypoints[0]["orientation"]

# ฟังก์ชันตรวจว่า YOLOP เพี้ยนหรือไม่
# เช่น กรณี lane detection หาย หรือขับเบี่ยงจากศูนย์กลางมาก

def is_yolop_reliable(center_da, image_width, lane_found, deviation_threshold=80):
    center_offset = abs(center_da - image_width // 2)
    return lane_found and center_offset <= deviation_threshold

def world_to_pixel(x, y, origin_x=0, origin_y=0, scale=20):
    """
    แปลงค่าพิกัดจริง (เมตร) เป็นตำแหน่งพิกเซลในภาพ BEV
    - origin_x, origin_y: จุดอ้างอิง (ตำแหน่งโลกที่ถือว่าเป็นพิกัด (0,0) ในภาพ)
    - scale: ความละเอียดของการแปลงเมตร → พิกเซล (ค่าทั่วไป 20-100)
    """
    px = int((x - origin_x) * scale)
    py = int((y - origin_y) * scale)
    return px, py

def draw_heading_on_bev(image, car_x, car_y, car_theta, origin_x=0, origin_y=0, scale=20, length=3.0):
    h, w = image.shape[:2]

    # แปลงพิกัดจริงเป็นพิกเซล
    start_px, start_py = world_to_pixel(car_x, car_y, origin_x, origin_y, scale)
    end_x = car_x + length * math.cos(car_theta)
    end_y = car_y + length * math.sin(car_theta)
    end_px, end_py = world_to_pixel(end_x, end_y, origin_x, origin_y, scale)

    print(f"🟢 Heading: Start({start_px}, {start_py}) → End({end_px}, {end_py}) | angle={math.degrees(car_theta):.1f}°")

    # เช็กว่าอยู่ในภาพ
    if 0 <= start_px < w and 0 <= start_py < h and 0 <= end_px < w and 0 <= end_py < h:
        cv2.arrowedLine(image, (start_px, start_py), (end_px, end_py), (0, 0, 255), 4, tipLength=0.2)
    else:
        print("❌ เส้นหลุดนอก BEV (หลังแปลงแล้ว)")
    
def draw_drivable_rows(bev_image, da_seg_bev_mask, num_lines=9, slope_threshold=0.05):
    """
    - วาดเส้นแนวนอนที่อยู่ในบริเวณ Drivable Area
    - ประมาณเส้นตรงจากจุดศูนย์กลาง
    - คืนค่า True/False ว่าควรขับตรงหรือไม่
    """
    height, width = da_seg_bev_mask.shape
    y_indices = np.linspace(-100, height - 1, num_lines).astype(int)
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

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-Front-latest4.mp4', fourcc, 30.0, (1280, 720))
out1 = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-BEV-latest4.mp4', fourcc, 30.0, (1280, 720))

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

        # ตรวจจับขอบเลนซ้ายสุด
        left_lane_pos = get_left_lane_position(ll_seg_mask)
        real_distance_to_lane = None  # กำหนดค่าเริ่มต้น
        # สร้าง BEV ก่อน
        bev_image = transform_to_bev(undistorted_frame.copy(), H)

        left_lane_history = []
        max_history = 5

        if left_lane_pos:
            left_lane_x, left_lane_y = left_lane_pos
            left_lane_bev = cv2.perspectiveTransform(np.array([[[left_lane_x, left_lane_y]]], dtype=np.float32), H)
            left_lane_x_bev = int(left_lane_bev[0][0][0])  # X ใน BEV
            left_lane_y_bev = int(left_lane_bev[0][0][1])  # Y ใน BEV

            real_distance_to_lane = float(pixel_to_real_distance(left_lane_x, left_lane_y, mtxL, R, T))
            
            # 🔹 แปลงพิกัดเลนซ้ายไปยัง BEV
            left_lane_bev = cv2.perspectiveTransform(np.array([[[left_lane_x, left_lane_y]]], dtype=np.float32), H)
            left_lane_x_bev, left_lane_y_bev = int(left_lane_bev[0][0][0]), int(left_lane_bev[0][0][1])

            cv2.imshow("Bird's Eye View", bev_image)  # อัปเดตภาพ
            print(f" ---> ระยะจากขอบเลนซ้าย (Real World): {real_distance_to_lane:.2f} เมตร")

        # ----------------- สร้าง BEV -----------------
        bev_image = transform_to_bev(undistorted_frame.copy(), H)

        # แปลงตำแหน่ง car_x, car_y → พิกเซล (อิงตามขนาดภาพ 1280x720 หรือ scaling เองก็ได้)
        px = int(car_x * 100)  # scale จากเมตร → พิกเซล
        py = int(car_y * 100)

        trajectory_points.append((px, py))

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
        # draw_heading_on_bev(bev_image, car_x, car_y, car_theta, origin_x=car_x - 6.4, origin_y=car_y - 3.6, scale=100)
        # 🟡 ตรวจสอบศูนย์กลางบนแนวขวางหลายจุด
        da_seg_mask_uint8 = (da_seg_mask > 0).astype(np.uint8) * 255
        da_seg_bev_mask = cv2.warpPerspective(da_seg_mask_uint8, H, (1280, 720))
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
        
        # ใช้ฟังก์ชันคำนวณ
        # ----------------- แสดงเส้น heading และเส้นเลนซ้ายบน BEV -----------------
        if left_lane_pos:
            left_lane_x, left_lane_y = left_lane_pos
            real_distance_to_lane = float(pixel_to_real_distance(left_lane_x, left_lane_y, mtxL, R, T))
            print(f" ---> ระยะจากขอบเลนซ้าย (Real World): {real_distance_to_lane:.2f} เมตร")
            print("📍 left_lane_pos =", left_lane_pos)

            # จุดเริ่มต้นของรถในพิกัดโลก
            start_world = np.array([car_x, car_y])
            heading_length = 2.0  # ความยาวเส้น heading (หน่วยเป็นเมตร)
            end_world = start_world + heading_length * np.array([np.cos(car_theta), np.sin(car_theta)])

            # แปลง world → pixel
            origin_x = car_x - 6.4  # กำหนด origin X ให้ตรงกับตำแหน่งของรถ
            origin_y = car_y - 3.6  # กำหนด origin Y ให้ตรงกับตำแหน่งของรถ
            scale = 100  # 1 เมตร = 100 พิกเซล

            def world_to_pixel(x, y, origin_x, origin_y, scale=100):
                px = int((x - origin_x) * scale)
                py = int((y - origin_y) * scale)
                return px, py

            start_px, start_py = world_to_pixel(start_world[0], start_world[1], origin_x, origin_y, scale)
            slope = np.tan(car_theta)  # คำนวณ slope จาก car_theta

            height = bev_image.shape[0]  # ความสูงของภาพ BEV

            # วาดเส้น heading เต็มจอ (สีเหลือง)
            # try:
            #     x0 = int(start_px - (start_py) / slope)      # y = 0
            #     x1 = int(start_px + (height - start_py) / slope)  # y = height
            #     cv2.line(bev_image, (x0, 0), (x1, height), (0, 255, 255), 2)
            # except ZeroDivisionError:
            #     # กรณีมุม 90° (ตรงพอดี)
            #     cv2.line(bev_image, (start_px, 0), (start_px, height), (0, 255, 255), 2)

            # วาดเส้นเลนซ้าย (สีฟ้า) แนวดิ่ง โดยแปลงพิกัดจาก world → pixel
            # เก็บประวัติเลนซ้าย
            left_lane_history.append(left_lane_x)
            if len(left_lane_history) > max_history:
                left_lane_history.pop(0)

            # ใช้ค่าเฉลี่ยของประวัติแทนค่าปัจจุบัน
            smooth_left_lane_x = int(np.mean(left_lane_history))
            lane_px, _ = world_to_pixel(smooth_left_lane_x, car_y, origin_x, origin_y, scale)
            cv2.line(bev_image, (lane_px, 0), (lane_px, bev_image.shape[0]), (255, 0, 0), 2)

            # (Optional) ใส่ข้อความช่วย debug
            distance_text = f"Lane Distance : {real_distance_to_lane:.2f} m"
            cv2.putText(bev_image, distance_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bev_image, f"MODE : {mode.upper()}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Bird's Eye View", bev_image)  # BEV มีแค่ Drivable Area & Lane (ไม่มี ROI และ Bounding Box)
            
            # ปรับทิศทางรถ
            if real_distance_to_lane > 0.8:
                print(" ---> อยู่ห่างเลนเกินไป → ขยับเข้าเลนซ้าย")
                steering_command = -1  # หมุนพวงมาลัยซ้าย
            elif real_distance_to_lane < 0.8:
                print(" ---> อยู่ใกล้เลนเกินไป → ขยับออกจากเลนซ้าย")
                steering_command = 1  # หมุนพวงมาลัยขวา
            else:
                print(" ---> ✅ ระยะโอเค → วิ่งตรงไป")
                steering_command = 0  # ตรงไป

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

        lane_found = left_lane_pos is not None
        center_da = get_center_of_drivable_area_bev(da_seg_mask)

        # ----------------- ตรวจสอบว่าโหมดเป็น YOLOP หรือ Pure Pursuit -----------------
        if is_yolop_reliable(center_da, bev_image.shape[1], lane_found):
            mode = "yolop"
            if real_distance_to_lane > 0.8:
                steering_command = -1
            elif real_distance_to_lane < 0.8:
                steering_command = 1
            else:
                steering_command = 0
            speed = 0.4
        else:
            mode = "pure_pursuit"
            wp = find_lookahead_point(car_x, car_y, waypoints, lookahead_distance=2.0)
            steering_angle, speed, dist = pure_pursuit_control(car_x, car_y, car_theta, wp)
            
            # พิมพ์ข้อมูลต่าง ๆ ในเทอร์มินอล
            print(f"🛣️ Pure Pursuit Mode")
            print(f"  Waypoint Index: {current_wp_index}")
            print(f"  Waypoint: x={wp['Position_X']}, y={wp['Position_Y']}")
            print(f"  Steering Angle δ (radians): {steering_angle:.3f} rad")
            print(f"  Distance to Waypoint: {dist:.2f} meters")
            print(f"  Speed: {speed:.2f} m/s")
            
            # อัปเดตตำแหน่งรถ (ใช้เมื่อไม่มี SLAM/GPS จริง)
            dt = 0.1  # timestep
            car_theta += steering_angle * dt  # คำนวณมุมของรถ
            car_theta = normalize_angle(car_theta)  # ปรับมุมให้เป็นระหว่าง -pi ถึง +pi
            car_x += speed * math.cos(car_theta) * dt  # คำนวณตำแหน่ง x
            car_y += speed * math.sin(car_theta) * dt  # คำนวณตำแหน่ง y

            if dist < 0.5 and current_wp_index < len(waypoints) - 1:
                current_wp_index += 1  # อัปเดต waypoint index

        # พิมพ์ข้อมูลตำแหน่งรถ
        print(f"🧠 Mode: {mode} | Speed: {speed:.2f} | Steering: {steering_angle if mode=='pure_pursuit' else steering_command} | WP Index: {current_wp_index}")

# ----------------- ปิดกล้อง -----------------
out.release()
out1.release()
zed.close()
cv2.destroyAllWindows()

