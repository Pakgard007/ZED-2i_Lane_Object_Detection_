import pyzed.sl as sl
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.utils import show_seg_result
from lib.core.general import non_max_suppression

# ---------------- 1. โหลดโมเดล YOLOP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")
model = get_net(cfg).to(device)
model.load_state_dict(torch.load("YOLOP/weights/End-to-end.pth", map_location=device)['state_dict'])
model.eval()

# Normalization ตาม YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ---------------- 2. ฟังก์ชันแปลงเป็น BEV ----------------
def IPM(image):
    height, width = image.shape[:2]
    
    param1 = 520
    param2 = 50
    
    original_points = np.float32([
        [0, (height // 2) + param2],         
        [width, (height // 2) + param2],      
        [width, height],          
        [0, height],               
    ])

    destination_points = np.float32([
        [150, 0],                    
        [width - 150, 0],               
        [width - param1, height * 2],    
        [param1, height * 2],            
    ])

    matrix = cv2.getPerspectiveTransform(original_points, destination_points)
    bev_image = cv2.warpPerspective(image, matrix, (width, height * 2))

    # Resize กลับเป็นขนาดเดิม
    bev_image = cv2.resize(bev_image, (width, height))
    
    return bev_image

# ---------------- 3. เปิดกล้อง ZED 2i ----------------
zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30, depth_mode=sl.DEPTH_MODE.NONE)

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้อง ZED 2i ได้!")
    exit()

image = sl.Mat()

# ---------------- 4. วนลูป อ่านภาพ และทำ YOLOP + BEV ----------------
while True:
    try:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ✅ Resize ภาพให้พอดีกับ YOLOP
            frame_resized = cv2.resize(frame, (640, 640))
            input_tensor = transform(frame_resized).unsqueeze(0).to(device)

            # ✅ ทำ Object Detection & Lane Detection ด้วย YOLOP
            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = model(input_tensor)

            # ✅ ใช้ argmax เลือกค่าที่มีโอกาสมากที่สุดในแต่ละพิกเซล
            da_seg_out = da_seg_out.sigmoid().cpu().numpy()
            ll_seg_out = ll_seg_out.sigmoid().cpu().numpy()

            da_seg_out = np.argmax(da_seg_out, axis=1).squeeze(0)
            ll_seg_out = np.argmax(ll_seg_out, axis=1).squeeze(0)

            # ✅ ใช้ show_seg_result() เพื่อรวมผลลัพธ์
            img_det = show_seg_result(frame_resized, (da_seg_out, ll_seg_out), index=0, epoch=0, is_demo=True)

            # ✅ แปลงภาพเป็น Bird’s Eye View (BEV)
            frame_bev = IPM(img_det)

            # ✅ ตรวจสอบ dtype และแปลงให้เป็น np.uint8
            if frame_bev.dtype != np.uint8:
                frame_bev = (frame_bev * 255).astype(np.uint8)

            if img_det.dtype != np.uint8:
                img_det = (img_det * 255).astype(np.uint8)

            # ✅ ปรับขนาดให้เท่ากัน
            if frame_bev.shape[:2] != img_det.shape[:2]:
                frame_bev = cv2.resize(frame_bev, (img_det.shape[1], img_det.shape[0]))

            # ✅ รวมภาพแบบ Side-by-Side
            frame_output = cv2.hconcat([img_det, frame_bev])

            # ✅ แสดงผล
            cv2.imshow("YOLOP & BEV (ZED 2i)", frame_output)

            # ✅ กด 'q' เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

# ---------------- 5. ปิดกล้องและปิดหน้าต่าง ----------------
zed.close()
cv2.destroyAllWindows()
