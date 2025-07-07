import argparse
import os, sys
import shutil
import time
from pathlib import Path
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import numpy as np
import pyzed.sl as sl
import torchvision.transforms as transforms
from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import show_seg_result

# ตั้งค่าการ Normalize สำหรับ YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ตั้งค่าอุปกรณ์ให้ใช้ CPU เท่านั้น
device = torch.device("cpu")

# ----------------- 1. เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NONE  

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image = sl.Mat()

# ----------------- 2. โหลดโมเดล YOLOP -----------------
model = get_net(cfg)
checkpoint = torch.load("YOLOP/weights/End-to-end.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device).eval()

print("✅ Model Loaded Successfully!")
print(model)  # แสดงโครงสร้างโมเดล

# ----------------- 3. ค่า Calibrate ของกล้อง ZED -----------------
K = np.array([[546.403, 0, 637.302],  
              [0, 546.565, 362.371],  
              [0, 0, 1]])
dist_coeffs = np.array([-0.057582, 0.033764, -0.000267, -0.000388, -0.014723])

# คำนวณ Homography Matrix โดยใช้ค่าคาลิเบรตที่ถูกต้อง
src_pts = np.float32([[400, 720], [880, 720], [500, 500], [800, 500]])
dst_pts = np.float32([[300, 900], [900, 900], [300, 500], [900, 500]])
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# ----------------- 4. อ่านภาพจากกล้อง, ทำ YOLOP และแปลง BEV -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, K, dist_coeffs)

        # Resize ภาพให้ตรงกับ input ของ YOLOP
        frame_resized = cv2.resize(undistorted_frame, (640, 640))

        # ทำให้เป็น Tensor สำหรับ YOLOP
        img_tensor = transform(frame_resized).unsqueeze(0).to(device).float()

        # รัน YOLOP
        with torch.no_grad():
            _, da_seg_out, ll_seg_out = model(img_tensor)

        # แปลงผล Lane Segmentation
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(720, 1280), mode='bilinear', align_corners=False)
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # แปลงผล Drivable Area Segmentation
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(720, 1280), mode='bilinear', align_corners=False)
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        # ตรวจสอบผลลัพธ์ที่ได้ (debugging)
        print("Lane Segmentation Unique Values:", np.unique(ll_seg_mask))
        print("Drivable Area Unique Values:", np.unique(da_seg_mask))

        # ตรวจสอบว่ามีข้อมูลจริง ไม่ใช่ค่าศูนย์ทั้งหมด
        if np.max(ll_seg_mask) == 0 and np.max(da_seg_mask) == 0:
            print("❌ YOLOP ไม่สามารถตรวจจับเลนหรือพื้นที่ขับขี่ได้")
            continue  # ข้ามการประมวลผลต่อไป

        # แปลงเป็น Bird's Eye View
        bev_image = cv2.warpPerspective(undistorted_frame, H, (1280, 720))  
        bev_lanes = cv2.warpPerspective(ll_seg_mask.astype(np.uint8) * 255, H, (1280, 720))
        bev_drivable_area = cv2.warpPerspective(da_seg_mask.astype(np.uint8) * 255, H, (1280, 720))

        # ผสม Lane และ Drivable Area บน BEV
        bev_combined = np.zeros((720, 1280, 3), dtype=np.uint8)  
        bev_combined[:, :, 1] = bev_drivable_area  # สีเขียวแสดงพื้นที่ขับขี่
        bev_combined[:, :, 2] = bev_lanes  # สีแดงแสดงเลนถนน

        # แปลงภาพให้อยู่ใน uint8 ก่อนแสดงผล
        ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
        da_seg_mask = (da_seg_mask * 255).astype(np.uint8)

        # แสดงผลลัพธ์
        cv2.imshow("Front View - ZED 2i", frame)
        cv2.imshow("Bird's Eye View", bev_image)
        cv2.imshow("Lane Segmentation BEV", bev_lanes)
        cv2.imshow("Drivable Area BEV", bev_drivable_area)
        cv2.imshow("Raw Lane Segmentation", ll_seg_mask)
        cv2.imshow("Raw Drivable Area", da_seg_mask)
        cv2.imshow("Combined BEV", bev_combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 5. ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()
