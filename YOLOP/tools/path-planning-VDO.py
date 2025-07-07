import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os, sys

# ---------------- 1️⃣ โหลดโมเดล YOLOP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.utils import show_seg_result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

model = get_net(cfg).to(device)
model.load_state_dict(torch.load("YOLOP/weights/End-to-end.pth", map_location=device)['state_dict'])
model.eval()

# ✅ Normalization ตาม YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ---------------- 2️⃣ โหลดค่าพารามิเตอร์ของกล้อง ----------------
calib_data = np.load("camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = np.array([-0.077, 0.067, 0.00015, -0.00006, -0.028])

extrinsic_data = np.load("extrinsic_parameters.npz")
R = extrinsic_data["R_left"]
T = extrinsic_data["T_left"]

# ✅ คำนวณ Homography Matrix โดยใช้การแมนนวลจุด
src_pts = np.array([[300, 700], [1000, 700], [550, 300], [750, 300]], dtype=np.float32)
dst_pts = np.array([[300, 850], [900, 850], [300, 400], [900, 400]], dtype=np.float32)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# ---------------- 3️⃣ เปิดวิดีโอ ----------------
video_path = "/home/mag/satoi/final.mp4"  # 🔹 เปลี่ยนเป็นพาธวิดีโอของคุณ
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดไฟล์วิดีโอได้!")
    exit()

# ---------------- 4️⃣ ฟังก์ชันสร้าง Occupancy Grid Map -----------------
def create_occupancy_grid(drivable_area):
    height, width = drivable_area.shape
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[drivable_area > 0] = 0  # พื้นที่ที่ขับได้ = 0 (ดำ)
    grid[drivable_area == 0] = 1  # สิ่งกีดขวาง = 1 (ขาว)
    return grid

# ---------------- 5️⃣ วนลูปอ่านเฟรมจากวิดีโอ และทำ YOLOP + BEV -----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🚀 วิดีโอจบแล้ว!")
        break

    try:
        # ✅ แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # ✅ Resize ภาพให้พอดีกับ YOLOP
        frame_resized = cv2.resize(undistorted_frame, (640, 640))
        input_tensor = transform(frame_resized).unsqueeze(0).to(device)

        # ✅ ทำ Object Detection & Lane Detection ด้วย YOLOP
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(input_tensor)

        # ✅ ใช้ argmax เพื่อเลือกพื้นที่ที่ขับได้ (ย้ายข้อมูลจาก GPU → CPU ก่อนใช้ NumPy)
        da_seg_out = da_seg_out.sigmoid().cpu().numpy()
        ll_seg_out = ll_seg_out.sigmoid().cpu().numpy()

        da_seg_out = np.argmax(da_seg_out, axis=1).squeeze(0)
        ll_seg_out = np.argmax(ll_seg_out, axis=1).squeeze(0)

        # ✅ Debug: ตรวจสอบค่า Drivable Area
        print(f"Drivable Area Unique Values: {np.unique(da_seg_out)}")
        print(f"Lane Line Unique Values: {np.unique(ll_seg_out)}")

        # ✅ แสดงผล YOLOP Front View
        img_det = show_seg_result(frame_resized, (da_seg_out, ll_seg_out), index=0, epoch=0, is_demo=True)
        cv2.imshow("Front View (YOLOP)", img_det)

        # ✅ แปลง Drivable Area เป็น BEV
        bev_drivable_area = cv2.warpPerspective(da_seg_out.astype(np.uint8) * 255, H, (1280, 720))

        # ✅ Debug: ตรวจสอบค่า BEV
        print(f"BEV Min: {bev_drivable_area.min()}, Max: {bev_drivable_area.max()}")

        # ✅ ป้องกัน BEV เป็นสีดำล้วน
        if bev_drivable_area.max() > 0:
            bev_drivable_area = cv2.normalize(bev_drivable_area, None, 0, 255, cv2.NORM_MINMAX)
            bev_drivable_area = bev_drivable_area.astype(np.uint8)

        cv2.imshow("Bird's Eye View (BEV)", bev_drivable_area)

        # ✅ แปลง BEV เป็น Occupancy Grid
        occupancy_grid = create_occupancy_grid(bev_drivable_area)

        # ✅ แปลง Grid เป็นภาพเพื่อแสดงผล
        occupancy_grid_vis = cv2.cvtColor(occupancy_grid * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Occupancy Grid Map", occupancy_grid_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

# ---------------- 6️⃣ ปิดวิดีโอ -----------------
cap.release()
cv2.destroyAllWindows()
