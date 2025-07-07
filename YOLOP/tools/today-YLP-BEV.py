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

    det_pred = non_max_suppression(det_out[0], conf_thres=0.25, iou_thres=0.45)

    print(f"✅ Detection Results - Drivable Area: {np.sum(da_seg_mask)}, Lanes: {np.sum(ll_seg_mask)}, Objects: {len(det_pred[0]) if det_pred else 0}")

    return da_seg_mask, ll_seg_mask, det_pred

def draw_bounding_boxes(image, detections):
    if detections is not None and len(detections):
        for det in detections:
            det[:, :4] = scale_coords(image.shape[:2], det[:, :4], image.shape).round()
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

        # Overlay ผลลัพธ์ลงบนภาพ
        overlay_da = overlay_segmentation(undistorted_frame, da_seg_mask, (0, 255, 0))
        overlay_ll = overlay_segmentation(overlay_da, ll_seg_mask, (0, 0, 255))
        overlay_ll = draw_bounding_boxes(overlay_ll, det_pred)

        # แปลงเป็น Bird's Eye View
        bev_image = transform_to_bev(overlay_ll, H)

        # ปรับขนาด BEV ให้ใหญ่ขึ้นก่อนแสดง
        large_bev = cv2.resize(bev_image, (1280, 720))
        cv2.imshow("Bird's Eye View", large_bev)

        # แสดงผลลัพธ์
        cv2.imshow("Front View - YOLOP", overlay_ll)
        # cv2.imshow("Bird's Eye View", bev_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()
