import os, sys
import cv2
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms
from lib.config import cfg
from lib.models import get_net

# ตั้งค่าการ Normalize สำหรับ YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ตั้งค่าอุปกรณ์ให้ใช้ CPU เท่านั้น
device = torch.device("cpu")

# โหลดโมเดล YOLOP
model = get_net(cfg)
checkpoint = torch.load("YOLOP/weights/End-to-end.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device).eval()
print("✅ Model Loaded Successfully!")

# เปิดวิดีโอ
video_path = "/home/satoi/final.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้!")
    exit()

# อ่านเฟรมแรกเพื่อหาขนาดของวิดีโอ
ret, frame = cap.read()
if not ret:
    print("❌ ไม่สามารถอ่านเฟรมแรกของวิดีโอได้!")
    cap.release()
    exit()

frame_height, frame_width, _ = frame.shape
print(f"📏 ขนาดวิดีโอ: {frame_width}x{frame_height}")

# ----------------- ปรับแต่งพิกัด BEV -----------------
top_view_offset = 100  # เพิ่มการดึงแนวตั้งให้มองจากบนลงมา

src_pts = np.float32([
    [frame_width * 0.15, frame_height * 0.95],  # มุมซ้ายล่าง
    [frame_width * 0.85, frame_height * 0.95],  # มุมขวาล่าง
    [frame_width * 0.35, frame_height * 0.6],  # มุมซ้ายบน
    [frame_width * 0.65, frame_height * 0.6]   # มุมขวาบน
])

dst_pts = np.float32([
    [250, 720 + top_view_offset],  # ซ้ายล่าง
    [1030, 720 + top_view_offset],  # ขวาล่าง
    [250, top_view_offset],  # ซ้ายบน
    [1030, top_view_offset]   # ขวาบน
])

# ใช้ Perspective Transform เพื่อให้ได้ BEV ที่ดูดีขึ้น
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# วนลูปอ่านเฟรมและประมวลผล YOLOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("📌 วิดีโอเล่นจบแล้ว!")
        break

    # แก้ Distortion
    undistorted_frame = cv2.undistort(frame, np.eye(3), np.zeros(5))

    # Resize ให้ YOLOP ใช้งานได้ดีขึ้น
    frame_resized = cv2.resize(undistorted_frame, (640, 640))

    # แปลงเป็น Tensor
    img_tensor = transform(frame_resized).unsqueeze(0).to(device).float()

    # รัน YOLOP
    with torch.no_grad():
        _, da_seg_out, ll_seg_out = model(img_tensor)

    # แปลงผลลัพธ์ของ YOLOP
    ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(frame_height, frame_width), mode='bilinear', align_corners=False)
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(frame_height, frame_width), mode='bilinear', align_corners=False)
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    # แปลงเป็น BEV
    bev_image = cv2.warpPerspective(undistorted_frame, H, (1280, 720))
    bev_lanes = cv2.warpPerspective(ll_seg_mask.astype(np.uint8) * 255, H, (1280, 720))
    bev_drivable_area = cv2.warpPerspective(da_seg_mask.astype(np.uint8) * 255, H, (1280, 720))

    # ผสม Lane และ Drivable Area บน BEV
    bev_combined = np.zeros((720, 1280, 3), dtype=np.uint8)
    bev_combined[:, :, 1] = bev_drivable_area  # สีเขียวแสดงพื้นที่ขับขี่
    bev_combined[:, :, 2] = bev_lanes  # สีแดงแสดงเลนถนน

    # แสดงผล
    cv2.imshow("Front View - Video", frame)
    cv2.imshow("Bird's Eye View", bev_image)
    cv2.imshow("Lane Segmentation BEV", bev_lanes)
    cv2.imshow("Drivable Area BEV", bev_drivable_area)
    cv2.imshow("Combined BEV", bev_combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดไฟล์วิดีโอ
cap.release()
cv2.destroyAllWindows()
