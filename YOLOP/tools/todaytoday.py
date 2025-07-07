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
    
# ----------------- แปลง Front View เป็น BEV -----------------
def transform_to_bev(image, H):
    bev = cv2.warpPerspective(image, H, (1280, 720))
    return bev

# ----------------- อ่านภาพจากกล้อง ZED 2i และตรวจจับ -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # ตรวจจับ Drivable Area และ Lane
        da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(undistorted_frame, model)
        
        # สร้าง ROI ที่ตรงกลางภาพ
        roi, roi_coords = create_roi(undistorted_frame)
        # ตรวจจับสิ่งกีดขวางใน ROI
        det_pred, roi_with_boxes = detect_objects_in_roi(roi, model)

        # Overlay ผลลัพธ์ลงบนภาพ
        overlay_da = overlay_segmentation(undistorted_frame, da_seg_mask, (0, 255, 0))
        overlay_ll = overlay_segmentation(overlay_da, ll_seg_mask, (0, 0, 255))
        overlay_ll = draw_bounding_boxes(overlay_ll, det_pred)

        # แปลงเป็น Bird's Eye View
        bev_image = transform_to_bev(overlay_ll, H)

        # ตรวจสอบว่ามีการตรวจจับวัตถุใน ROI หรือไม่
        object_detected = len(det_pred[0]) > 0 if det_pred else False

        # การตัดสินใจหยุดหรือให้เดินหน้าต่อ
        decision = decision_making(object_detected)

        # พิมพ์ผลในเทอร์มินอล
        print(f"✅ Detection Results - Drivable Area: {np.sum(da_seg_mask)}, Lanes: {np.sum(ll_seg_mask)}, Objects: {len(det_pred[0]) if det_pred else 0}")
        print(decision)

        # แสดงผลลัพธ์ใน Front View โดยแสดง ROI
        cv2.imshow("Front View - YOLOP", overlay_ll)
        cv2.imshow("ROI", roi)  # แสดงแค่ใน Front View เท่านั้น

        # แสดงข้อมูลในเทอร์มินอล
        print(f"Drivable Area: {np.sum(da_seg_mask)}")
        print(f"Lanes: {np.sum(ll_seg_mask)}")
        print(f"Objects: {len(det_pred[0]) if det_pred else 0}")

        # แสดงผล BEV
        large_bev = cv2.resize(bev_image, (1280, 720))
        cv2.imshow("Bird's Eye View", large_bev)

        # รอให้กดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()