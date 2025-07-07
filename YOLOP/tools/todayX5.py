import argparse
import os, sys
import time
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import pyzed.sl as sl
import torch.nn.functional as F

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
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # ใช้ Depth Mode
init_params.coordinate_units = sl.UNIT.METER  # หน่วยเป็นเมตร  

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image = sl.Mat()

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
def create_roi(image, roi_width_pixels=300, roi_height_pixels=700):
    """
    สร้าง ROI ที่เป็นสี่เหลี่ยมผืนผ้าแนวตั้งตรงกลางของภาพในพิกเซล
    roi_width_pixels: ความกว้างของ ROI ที่ต้องการในพิกเซล
    roi_height_pixels: ความสูงของ ROI ที่ต้องการในพิกเซล
    """
    height, width = image.shape[:2]

    # คำนวณจุดเริ่มต้นของ ROI
    x_start = (width - roi_width_pixels) // 2
    y_start = (height - roi_height_pixels) // 2

    # สร้าง ROI
    roi = image[y_start:y_start + roi_height_pixels, x_start:x_start + roi_width_pixels]

    # วาดกรอบ ROI ในภาพ Front View (ไม่แสดงใน BEV)
    cv2.rectangle(image, (x_start, y_start), (x_start + roi_width_pixels, y_start + roi_height_pixels), (0, 255, 255), 2)

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
def get_depth_for_detections(depth_map, detections):
    """
    ดึงค่าความลึก (Depth) ของวัตถุที่ตรวจจับได้จาก Bounding Box
    """
    object_depths = []
    
    if detections is None or len(detections) == 0:
        return object_depths  # ถ้าไม่มีวัตถุ

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  # ค่าพิกัด Bounding Box

        # ตัด ROI ที่เป็น Bounding Box
        obj_depth_roi = depth_map[y1:y2, x1:x2]

        # ลบค่าที่เป็น NaN หรือ Inf ออก
        valid_depth = obj_depth_roi[np.isfinite(obj_depth_roi)]

        if len(valid_depth) == 0:
            object_depths.append(None)  # ถ้าไม่มีค่าที่ใช้ได้
        else:
            object_depths.append(np.median(valid_depth))  # ใช้ค่า median เพื่อลด Noise
    
    return object_depths

# ----------------- ตรวจจับ Drivable Area และ Lane -----------------
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

    # ใช้ non_max_suppression กับ det_out
    det_pred = non_max_suppression(det_out[0], conf_thres=0.25, iou_thres=0.45)

    # ตรวจสอบผลลัพธ์จาก non_max_suppression
    return da_seg_mask, ll_seg_mask, det_pred


# ----------------- ฟังก์ชันตรวจจับวัตถุใน ROI -----------------
def detect_objects_in_roi(roi, model):
    # ตรวจจับวัตถุใน ROI โดยใช้ฟังก์ชัน detect_obstacles ที่มีอยู่
    da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(roi, model)

    # ตรวจสอบว่า det_pred เป็น tensor หรือไม่
    if isinstance(det_pred, torch.Tensor):  # ถ้าเป็น tensor
        det_pred = det_pred.cpu().numpy()  # ย้ายไปที่ CPU และแปลงเป็น numpy array

    # ซ้อน Bounding Boxes ใน ROI
    roi_with_boxes = draw_bounding_boxes(roi, det_pred)

    return det_pred, roi_with_boxes

# ----------------- การตัดสินใจหยุดการเคลื่อนที่ -----------------
def decision_making(object_detected):
    if object_detected:
        return "🚫 STOP: Obstacle Ahead!"  # หากพบวัตถุใน ROI
    else:
        return "✅ SAFE: Continue Moving"  # ถ้าไม่มีการตรวจจับวัตถุ


def draw_bounding_boxes(image, detections):
    if detections is not None and len(detections):
        for det in detections:
            # ตรวจสอบว่า det เป็น numpy array หรือไม่
            if isinstance(det, torch.Tensor):
                det = det.cpu().numpy()  # ถ้าเป็น tensor ให้ย้ายไปที่ CPU และแปลงเป็น numpy array

            det = np.array(det)  # แปลงเป็น numpy array
            det_tensor = torch.tensor(det[:, :4].astype(float), dtype=torch.float32)

            # ใช้ scale_coords กับ Tensor
            det_tensor = scale_coords(image.shape[:2], det_tensor, image.shape).round()

            # แปลงกลับเป็น numpy.ndarray
            det[:, :4] = det_tensor.numpy().astype(int)

            for *xyxy, conf, cls in reversed(det):
                label = f'Obj {int(cls)} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(0, 0, 255), line_thickness=2)
    return image


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

def draw_distance_line(image, left_lane_x, real_distance_to_lane):
    """
    วาดเส้นที่แสดงระยะห่างจากขอบเลนใน Front View โดยอ้างอิงจากระยะทางจริง
    """
    car_x = image.shape[1] // 2  # ตำแหน่งกึ่งกลางของรถ (กลางภาพ)
    
    # ใช้ค่าคงที่ในการสเกล (แปลงเมตรเป็นพิกเซลโดยคร่าว ๆ)
    scale_factor = 100  # ค่าคงที่ที่กำหนด (สามารถปรับเพื่อให้เหมาะสมกับภาพ)

    # คำนวณตำแหน่งที่จะวาดเส้น โดยใช้ระยะทางจริง
    line_start = (car_x, image.shape[0])  # จุดเริ่มต้นที่ตำแหน่งรถ (ด้านล่างของภาพ)
    line_end_x = int(car_x + (real_distance_to_lane * scale_factor))  # ใช้ scale_factor แทน pixels_per_meter
    line_end = (line_end_x, 0)  # เส้นจะไปถึงขอบด้านบน

    # สีของเส้นและความหนา
    color = (0, 255, 255)  # สีเหลือง
    thickness = 2

    # วาดเส้น
    # cv2.line(image, line_start, line_end, color, thickness)
    
    return image

# ----------------- แปลง Front View เป็น BEV -----------------
def transform_to_bev(image, H):
    bev = cv2.warpPerspective(image, H, (1280, 720))
    return bev


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


def track_drivable_area(da_seg_mask, prev_center=None):
    # คำนวณจุดศูนย์กลางของ Drivable Area
    y, x = np.where(da_seg_mask > 0)
    
    if len(x) == 0 or len(y) == 0:  # หากไม่พบพื้นที่ขับได้
        return None, prev_center
    
    # คำนวณตำแหน่งศูนย์กลาง
    center_x = np.mean(x)
    center_y = np.mean(y)
    
    # ถ้ามีข้อมูลศูนย์กลางก่อนหน้านี้ (ในเฟรมก่อนหน้า), คำนวณการเคลื่อนที่
    if prev_center is not None:
        movement_x = center_x - prev_center[0]
        movement_y = center_y - prev_center[1]
        movement = np.sqrt(movement_x**2 + movement_y**2)
        # print(f"การเคลื่อนที่: {movement:.2f} pixels")
    
    # ส่งค่ากลับ (ศูนย์กลางในปัจจุบัน, ศูนย์กลางในเฟรมก่อนหน้า)
    return (center_x, center_y), (center_x, center_y)


def draw_distance_line_on_bev(bev_image, left_lane_x, car_x, real_distance_to_lane):
    """
    วาดเส้นแสดงระยะห่างจากขอบเลนในมุมมอง BEV
    """
    scale_factor = 100  # แปลงเมตรเป็นพิกเซลโดยประมาณ (ค่าปรับได้)
    
    # คำนวณตำแหน่งที่จะวาดเส้น
    line_start = (car_x, bev_image.shape[0])  # จุดเริ่มต้นที่ตำแหน่งรถ (ด้านล่างของภาพ)
    line_end_x = int(car_x + (real_distance_to_lane * scale_factor))  # ใช้ scale_factor คำนวณ
    line_end = (line_end_x, 0)  # จุดสิ้นสุดของเส้น
    
    return bev_image

def draw_distance_line_in_bev(bev_image, left_lane_x, distance_to_lane):
    car_x = bev_image.shape[1] // 2  # จุดกึ่งกลางของรถใน BEV
    car_y = bev_image.shape[0] - 50  # ให้เส้นเริ่มจากตำแหน่งล่างของ BEV

    # คำนวณจุดปลายของเส้น
    line_start = (car_x, car_y)  
    line_end_x = int(left_lane_x)  
    line_end_y = int(car_y - (distance_to_lane * 100))  # ปรับให้สัมพันธ์กับระยะห่างจริง

    # ป้องกันค่าพิกัดออกนอกภาพ
    line_end_x = max(0, min(bev_image.shape[1] - 1, line_end_x))
    line_end_y = max(0, min(bev_image.shape[0] - 1, line_end_y))

    return bev_image

def add_text_to_bev(bev_image, distance_to_lane):
    """ เพิ่มข้อความแสดงสถานะการตรวจจับระยะห่างจากขอบเลนใน BEV """
    if distance_to_lane > 1.0:
        status_text = "❗ Far from Lane! Move Left"
        color = (0, 0, 255)  # สีแดง
    elif distance_to_lane < 1.0:
        status_text = "❗ Too Close to Lane! Move Right"
        color = (0, 0, 255)  # สีแดง
    else:
        status_text = "✅ Good Distance"
        color = (0, 255, 0)  # สีเขียว

    # เพิ่มข้อความลงในภาพ BEV
    cv2.putText(bev_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return bev_image

def add_text_to_front_view(front_view_image, distance_to_lane):
    """ เพิ่มข้อความแสดงสถานะการตรวจจับระยะห่างจากขอบเลนใน Front View """
    if distance_to_lane is None:
        return front_view_image  # ถ้าไม่มีข้อมูล ให้คืนค่าเดิมโดยไม่เพิ่มข้อความ
    
    if distance_to_lane > 1.0:
        status_text = "--> Far from Lane! Move Left"
        color = (0, 0, 255)  # สีแดง
    elif distance_to_lane < 1.0:
        status_text = "--> Too Close to Lane! Move Right"
        color = (0, 0, 255)  # สีแดง
    else:
        status_text = "--> Good Distance"
        color = (0, 255, 0)  # สีเขียว

    # เพิ่มข้อความลงในภาพ Front View
    # cv2.putText(front_view_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return front_view_image

def get_center_of_drivable_area_bev(da_seg_bev):
    """
    คำนวณตำแหน่งศูนย์กลางของ Drivable Area ในมุมมอง BEV
    """
    height, width = da_seg_bev.shape
    y, x = np.where(da_seg_bev > 0)  # หาพิกเซลที่เป็นพื้นที่ขับขี่

    if len(x) == 0:  # ถ้าไม่มีข้อมูล
        return width // 2  # คืนค่ากึ่งกลางของจอ

    return int(np.mean(x))  # คำนวณค่ากลางของ Drivable Area ใน BEV


def is_driving_straight(center_da, center_ref, threshold=20):
    """
    ตรวจสอบว่า Drivable Area อยู่ตรงกลางภาพ BEV หรือไม่
    ถ้าอยู่ใน threshold ที่กำหนด → ขับตรงได้เลย
    """
    offset = abs(center_da - center_ref)  # หาความต่างระหว่างจุดศูนย์กลางของ DA กับ กึ่งกลางภาพ BEV

    if offset <= threshold:
        return "✅ Stay Straight"
    else:
        return "✅ Keep Going (Slight Deviation OK)"

def highlight_left_lane(bev_image, left_lane_x, left_lane_y):
    """ วาดสัญลักษณ์แสดงขอบเลนซ้ายที่ใช้วัดระยะ """
    if left_lane_x is not None and left_lane_y is not None:
        # วาดวงกลมสีน้ำเงินที่ขอบเลนซ้าย
        cv2.circle(bev_image, (left_lane_x, left_lane_y), 15, (0, 0, 255), -1)
        
        # วาดเส้นไขว้ X แสดงจุดอ้างอิง
        cv2.line(bev_image, (left_lane_x - 10, left_lane_y - 10), (left_lane_x + 10, left_lane_y + 10), (255, 255, 255), 2)
        cv2.line(bev_image, (left_lane_x - 10, left_lane_y + 10), (left_lane_x + 10, left_lane_y - 10), (255, 255, 255), 2)
        
    return bev_image

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-Front2.mp4', fourcc, 30.0, (1280, 720))
out1 = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-BEV2.mp4', fourcc, 30.0, (1280, 720))

# ----------------- อ่านภาพจากกล้อง ZED 2i และตรวจจับ -----------------
prev_center = None
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # ตรวจจับ Drivable Area และ Lane
        da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(undistorted_frame, model)
        
        # แทรคตำแหน่ง Drivable Area
        center, prev_center = track_drivable_area(da_seg_mask, prev_center)

        if center is not None:
            # วาดจุดศูนย์กลางของ Drivable Area
            cv2.circle(undistorted_frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)

        # ตรวจจับขอบเลนซ้ายสุด
        left_lane_pos = get_left_lane_position(ll_seg_mask)
        real_distance_to_lane = None  # กำหนดค่าเริ่มต้น
        # สร้าง BEV ก่อน
        bev_image = transform_to_bev(undistorted_frame.copy(), H)

        if left_lane_pos:
            left_lane_x, left_lane_y = left_lane_pos
            real_distance_to_lane = float(pixel_to_real_distance(left_lane_x, left_lane_y, mtxL, R, T))
            
            # 🔹 แปลงพิกัดเลนซ้ายไปยัง BEV
            left_lane_bev = cv2.perspectiveTransform(np.array([[[left_lane_x, left_lane_y]]], dtype=np.float32), H)
            left_lane_x_bev, left_lane_y_bev = int(left_lane_bev[0][0][0]), int(left_lane_bev[0][0][1])

            # 🔹 วาดเส้นแนวเดียวกับเลนที่เจอ
            cv2.line(bev_image, (left_lane_x_bev, left_lane_y_bev), (left_lane_x_bev, bev_image.shape[0]), (255, 0, 0), 3)

            # 🔹 วาดเส้นแสดงระยะห่างใน **BEV**
            bev_image = draw_distance_line_in_bev(bev_image, left_lane_x, real_distance_to_lane)
            cv2.imshow("Bird's Eye View", bev_image)  # อัปเดตภาพ
            # 🔹 เพิ่มข้อความแสดงค่าระยะห่างใน BEV
            bev_image = add_text_to_bev(bev_image, real_distance_to_lane)

            print(f" ---> ระยะจากขอบเลนซ้าย (Real World): {real_distance_to_lane:.2f} เมตร")


        left_lane_bev = cv2.perspectiveTransform(np.array([[[left_lane_x, left_lane_y]]], dtype=np.float32), H)
        left_lane_x_bev = int(left_lane_bev[0][0][0])  # X ใน BEV
        left_lane_y_bev = int(left_lane_bev[0][0][1])  # Y ใน BEV

        # ----------------- สร้าง BEV -----------------
        bev_image = transform_to_bev(undistorted_frame.copy(), H)

        # ----------------- ตรวจสอบว่า Drivable Area อยู่ตรงกลางหรือไม่ -----------------
        center_da = get_center_of_drivable_area_bev(da_seg_mask)
        driving_status = is_driving_straight(center_da, bev_image.shape[1] // 2)

        # ----------------- แสดงข้อความใน Terminal -----------------
        print(f"🚗 Driving Status: {driving_status}")

        # ----------------- เพิ่มข้อความลงบน BEV -----------------
        cv2.putText(bev_image, driving_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # ----------------- ตรวจสอบตำแหน่งขอบเลนซ้าย -----------------
        if left_lane_pos:
            left_lane_x, _ = left_lane_pos
            car_x = bev_image.shape[1] // 2  # กึ่งกลางของรถใน BEV

            # วาดเส้นระยะห่างใน BEV
            bev_image = draw_distance_line_on_bev(bev_image, left_lane_x, car_x, real_distance_to_lane)
            bev_image = highlight_left_lane(bev_image, left_lane_x, left_lane_y)  # วาดสัญลักษณ์แสดงขอบเลนซ้าย

        # ----------------- แสดงผล BEV -----------------
        cv2.imshow("Bird's Eye View", bev_image)  # แสดงผล BEV ที่มีการตรวจจับและข้อความ

        # ----------------- Overlay Drivable Area & Lane บน Front View -----------------
        overlay_da = overlay_segmentation(undistorted_frame.copy(), da_seg_mask, (0, 255, 0))  # ซ้อนสีเขียว (Drivable Area)
        overlay_ll = overlay_segmentation(overlay_da, ll_seg_mask, (0, 0, 255))  # ซ้อนสีแดง (Lane)

        # ----------------- สร้าง BEV โดยใช้ overlay_ll (ไม่มี ROI) -----------------
        bev_image = transform_to_bev(overlay_ll.copy(), H)  # ใช้ overlay_ll ที่ไม่มี ROI

        # ----------------- สร้าง ROI (เฉพาะ Front View เท่านั้น) -----------------
        roi, roi_coords = create_roi(overlay_ll)  # ใช้ overlay_ll เพื่อให้ ROI อยู่ที่ Front View เท่านั้น
        det_pred, roi_with_boxes = detect_objects_in_roi(roi, model)

        # ----------------- วาด ROI และ Bounding Box บน Front View -----------------
        front_view_with_roi = show_roi_in_front_view(overlay_ll, roi, roi_coords)  # ใช้ overlay_ll เพื่อให้มี Drivable Area & Lane
        front_view_with_roi = draw_bounding_boxes(front_view_with_roi, det_pred)  # วาด Bounding Box ใน Front View เท่านั้น


        # ----------------- เพิ่มข้อความใน Front View -----------------
        if real_distance_to_lane is not None:
            front_view_with_distance = draw_distance_line(front_view_with_roi, left_lane_x, real_distance_to_lane)  
        else:
            front_view_with_distance = front_view_with_roi  # ใช้ภาพเดิมถ้าไม่มีค่า


        if real_distance_to_lane is not None:
            front_view_with_text = add_text_to_front_view(front_view_with_distance, real_distance_to_lane)
        else:
            front_view_with_text = front_view_with_distance  # ถ้าไม่มีข้อมูล ใช้ภาพเดิม

        cv2.imshow("Front View - YOLOP", front_view_with_text)

        # ----------------- แสดงผล Front View -----------------
        cv2.imshow("Front View - YOLOP", front_view_with_distance)  # แสดงผลพร้อมเส้นระยะห่าง
        out.write(front_view_with_distance)
        cv2.imshow("Bird's Eye View", bev_image)  # BEV มีแค่ Drivable Area & Lane (ไม่มี ROI และ Bounding Box)
        out1.write(bev_image)


        # ----------------- พิมพ์ข้อมูลการตรวจจับ -----------------
        object_detected = len(det_pred[0]) > 0 if det_pred else False
        decision = decision_making(object_detected)
        print(f"✅ Detection Results - Drivable Area: {np.sum(da_seg_mask)}, Lanes: {np.sum(ll_seg_mask)}, Objects: {len(det_pred[0]) if det_pred else 0}")
        print(decision)

        # ใช้ฟังก์ชันคำนวณ
        if left_lane_pos:
            left_lane_x, _ = left_lane_pos
            real_distance_to_lane = float(pixel_to_real_distance(left_lane_x, left_lane_y, mtxL, R, T))
            print(f" ---> ระยะจากขอบเลนซ้าย (Real World): {float(real_distance_to_lane):.2f} เมตร")

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

# ----------------- ปิดกล้อง -----------------
out.release()
out1.release()
zed.close()
cv2.destroyAllWindows()

