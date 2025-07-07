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
init_params.camera_fps = 60
init_params.depth_mode = sl.DEPTH_MODE.NONE  

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
    cv2.rectangle(image, (x_start, y_start), (x_start + roi_width_pixels, y_start + roi_height_pixels), (0, 255, 0), 2)

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
model = load_yolop_model("/home/mag/satoi/python/YOLOP/weights/End-to-end.pth", device)

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
    cv2.rectangle(image, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (0, 255, 0), 2)
    return image

def draw_distance_line(image, left_lane_x, distance_to_lane, pixels_per_meter):
    """ วาดเส้นที่แสดงระยะห่างจากขอบเลนใน Front View """
    car_x = 640  # ตำแหน่งรถอยู่ที่กึ่งกลางใน BEV (แก้ไขตามตำแหน่งที่แท้จริง)
    
    # คำนวณตำแหน่งที่จะวาดเส้น
    line_start = (car_x, image.shape[0])  # จุดเริ่มต้นที่ตำแหน่งรถ (ด้านล่าง)
    line_end_x = int(car_x + (distance_to_lane * pixels_per_meter))
    line_end = (line_end_x, 0)  # เส้นจะไปถึงขอบด้านบน

    # สีของเส้นและความหนา
    color = (0, 255, 255)  # สีเหลือง
    thickness = 2
    
    # วาดเส้น
    cv2.line(image, line_start, line_end, color, thickness)
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

def compute_distance_to_lane(left_lane_x, pixels_per_meter):
    """ คำนวณระยะห่างจากรถไปยังขอบเลนซ้าย """
    car_x = 640  # กึ่งกลางภาพ BEV (ตำแหน่งรถ)
    pixel_distance = abs(car_x - left_lane_x)  # ระยะพิกเซลจากรถไปเลนซ้าย
    real_distance = pixel_distance / pixels_per_meter  # แปลงเป็นเมตร
    return real_distance

def get_left_lane_position(ll_seg_mask):
    """ หาเส้นขอบเลนซ้ายสุดในภาพ BEV """
    lane_pixels = np.where(ll_seg_mask > 0)  # หาจุดที่เป็นเลน
    if len(lane_pixels[0]) == 0:
        return None  # ถ้าไม่มีเลนซ้ายให้คืนค่า None --------x

    # หาจุดที่ใกล้กับกึ่งกลางภาพที่สุด
    lane_x = lane_pixels[1]  # ค่า x ของจุดเลน
    lane_y = lane_pixels[0]  # ค่า y ของจุดเลน
    center_x = ll_seg_mask.shape[1] // 2  # กึ่งกลางภาพในแนว x

    # หาจุดเลนซ้ายที่ใกล้ center_x ที่สุด
    left_lane_x = np.min(lane_x)  # ค่าต่ำสุดของ x คือขอบเลนซ้าย
    left_lane_y = lane_y[np.argmin(lane_x)]  # y ที่ตรงกับค่า x ซ้ายสุด
    return (left_lane_x, left_lane_y)

def compute_distance_to_lane(left_lane_x, pixels_per_meter):
    """ คำนวณระยะห่างจากรถไปยังขอบเลนซ้าย """
    car_x = 640  # กึ่งกลางภาพ BEV (ตำแหน่งรถ)
    pixel_distance = abs(car_x - left_lane_x)  # ระยะพิกเซลจากรถไปเลนซ้าย
    real_distance = pixel_distance / pixels_per_meter  # แปลงเป็นเมตร
    return real_distance

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
        print(f"การเคลื่อนที่: {movement:.2f} pixels")
    
    # ส่งค่ากลับ (ศูนย์กลางในปัจจุบัน, ศูนย์กลางในเฟรมก่อนหน้า)
    return (center_x, center_y), (center_x, center_y)

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
            cv2.circle(undistorted_frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

        # ใช้ฟังก์ชันคำนวณ
        left_lane_pos = get_left_lane_position(ll_seg_mask)
        if left_lane_pos:
            left_lane_x, _ = left_lane_pos
            pixels_per_meter = 50  # ค่านี้ต้องคำนวณจากการคาลิเบรตจริง
            distance_to_lane = compute_distance_to_lane(left_lane_x, pixels_per_meter)

            print(f" ---> ระยะจากขอบเลนซ้าย: {distance_to_lane:.2f} เมตร")

            # ปรับทิศทางรถ
            if distance_to_lane > 1.0:
                print(" ---> อยู่ห่างเลนเกินไป → ขยับเข้าเลนซ้าย")
                steering_command = -1  # หมุนพวงมาลัยซ้าย
            elif distance_to_lane < 1.0:
                print(" ---> อยู่ใกล้เลนเกินไป → ขยับออกจากเลนซ้าย")
                steering_command = 1  # หมุนพวงมาลัยขวา
            else:
                print(" ---> ✅ ระยะโอเค → วิ่งตรงไป")
                steering_command = 0  # ตรงไป

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

        # ----------------- วาดเส้นระยะห่างจากขอบเลนใน Front View -----------------
        left_lane_pos = get_left_lane_position(ll_seg_mask)
        if left_lane_pos:
            left_lane_x, _ = left_lane_pos
            pixels_per_meter = 50  # ค่านี้ต้องคำนวณจากการคาลิเบรตจริง
            distance_to_lane = compute_distance_to_lane(left_lane_x, pixels_per_meter)
            front_view_with_distance = draw_distance_line(front_view_with_roi, left_lane_x, distance_to_lane, pixels_per_meter)

        # ----------------- แสดงผล -----------------
        cv2.imshow("Front View - YOLOP", front_view_with_distance)  # แสดงผลพร้อมเส้นระยะห่าง
        cv2.imshow("Bird's Eye View", bev_image)  # BEV มีแค่ Drivable Area & Lane (ไม่มี ROI และ Bounding Box)

        # ----------------- พิมพ์ข้อมูลการตรวจจับ -----------------
        object_detected = len(det_pred[0]) > 0 if det_pred else False
        decision = decision_making(object_detected)
        print(f"✅ Detection Results - Drivable Area: {np.sum(da_seg_mask)}, Lanes: {np.sum(ll_seg_mask)}, Objects: {len(det_pred[0]) if det_pred else 0}")
        print(decision)

        # ใช้ฟังก์ชันคำนวณ
        left_lane_pos = get_left_lane_position(ll_seg_mask)
        if left_lane_pos:
            left_lane_x, _ = left_lane_pos
            pixels_per_meter = 50  # ค่านี้ต้องคำนวณจากการคาลิเบรตจริง
            distance_to_lane = compute_distance_to_lane(left_lane_x, pixels_per_meter)

            print(f" ---> ระยะจากขอบเลนซ้าย: {distance_to_lane:.2f} เมตร")

            # ปรับทิศทางรถ
            if distance_to_lane > 1.0:
                print(" ---> อยู่ห่างเลนเกินไป → ขยับเข้าเลนซ้าย")
                steering_command = -1  # หมุนพวงมาลัยซ้าย
            elif distance_to_lane < 1.0:
                print(" ---> อยู่ใกล้เลนเกินไป → ขยับออกจากเลนซ้าย")
                steering_command = 1  # หมุนพวงมาลัยขวา
            else:
                print(" ---> ✅ ระยะโอเค → วิ่งตรงไป")
                steering_command = 0  # ตรงไป

                
        # แสดงผลลัพธ์ใน Front View
        cv2.imshow("Front View - YOLOP", front_view_with_distance)
        cv2.imshow("ROI", roi)  # แสดงแค่ใน Front View เท่านั้น

        # แสดงผล BEV
        large_bev = cv2.resize(bev_image, (1280, 720))
        cv2.imshow("Bird's Eye View", large_bev)

        # รอให้กดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()

