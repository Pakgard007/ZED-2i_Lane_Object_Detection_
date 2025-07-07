import pyzed.sl as sl
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# YOLOP Modules
from lib.config import cfg
from lib.models import get_net
from lib.utils import show_seg_result
from lib.core.general import non_max_suppression

# ----------------- 1️⃣ โหลดโมเดล YOLOP -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")
model = get_net(cfg).to(device)
model.load_state_dict(torch.load("YOLOP/weights/End-to-end.pth", map_location=device)['state_dict'])
model.eval()

# Normalization ตาม YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ----------------- 2️⃣ เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30, depth_mode=sl.DEPTH_MODE.NONE)
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()
image = sl.Mat()

# ----------------- 3️⃣ โหลดค่าคาลิเบรต & Homography Matrix -----------------
calib_data = np.load("camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = np.array([
    -0.07709319689152208,  # k1
     0.06756180189133752,  # k2
     0.00015006759935512075,  # p1 (tangential distortion)
    -6.006342505065124e-05,  # p2 (tangential distortion)
    -0.028545020615709165  # k3
])

# กำหนดจุดสำหรับ Homography Matrix (ถ้าคำนวณอัตโนมัติผิดพลาด)
src_pts = np.array([[300, 600], [1000, 600], [500, 300], [800, 300]], dtype=np.float32)
dst_pts = np.array([[300, 850], [900, 850], [300, 400], [900, 400]], dtype=np.float32)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# ----------------- 4️⃣ วนลูป อ่านภาพ และทำ YOLOP + BEV -----------------
while True:
    try:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 🔹 แก้ Distortion
            undistorted_frame = cv2.undistort(frame, mtxL, distL)

            # 🔹 Resize ภาพให้เหมาะกับ YOLOP
            frame_resized = cv2.resize(undistorted_frame, (640, 640))
            input_tensor = transform(frame_resized).unsqueeze(0).to(device)

            # 🔹 ทำ Object Detection & Lane Detection ด้วย YOLOP
            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = model(input_tensor)

            # 🔹 ใช้ argmax เลือกค่าที่มีโอกาสมากที่สุดในแต่ละพิกเซล
            da_seg_out = np.argmax(da_seg_out.sigmoid().cpu().numpy(), axis=1).squeeze(0)
            ll_seg_out = np.argmax(ll_seg_out.sigmoid().cpu().numpy(), axis=1).squeeze(0)

            # 🔹 ใช้ show_seg_result() เพื่อรวมผลลัพธ์
            img_det = show_seg_result(frame_resized, (da_seg_out, ll_seg_out), index=0, epoch=0, is_demo=True)

            # 🔹 แปลงภาพเป็น Bird’s Eye View (BEV)
            bev_image = cv2.warpPerspective(img_det, H, (1280, 720))

            # 🔹 ปรับขนาดภาพเพื่อแสดงผล
            front_resized = cv2.resize(img_det, (640, 720))
            bev_resized = cv2.resize(bev_image, (640, 720))

            # 🔹 รวมภาพแบบ Side-by-Side
            frame_output = cv2.hconcat([front_resized, bev_resized])

            # 🔹 แสดงผล
            cv2.imshow("YOLOP + BEV (ZED 2i)", frame_output)

            # 🔹 กด 'q' เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

# ----------------- 5️⃣ ปิดกล้องและปิดหน้าต่าง -----------------
zed.close()
cv2.destroyAllWindows()
