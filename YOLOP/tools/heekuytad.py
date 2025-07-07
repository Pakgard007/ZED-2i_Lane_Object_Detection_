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
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # ใช้ Depth Mode
init_params.coordinate_units = sl.UNIT.METER  # หน่วยเป็นเมตร

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image = sl.Mat()
depth_map = sl.Mat()

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

    return da_seg_mask, ll_seg_mask, det_pred

# ----------------- อ่านภาพจากกล้อง ZED 2i และตรวจจับ -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # ตรวจจับ Drivable Area และ Lane
        da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(frame, model)

        # หาค่า Depth ของวัตถุที่ตรวจจับได้
        object_depths = get_depth_for_detections(depth_map.get_data(), det_pred[0])

        # วาด Bounding Box และแสดงหมายเลขวัตถุ + ระยะห่าง
        if det_pred[0] is not None:
            for i, det in enumerate(det_pred[0]):
                x1, y1, x2, y2, conf, cls = det
                obj_id = i + 1  # หมายเลขวัตถุ
                label = f'Object {obj_id}'

                # แสดงระยะห่างของวัตถุ
                if object_depths[i] is not None:
                    depth_text = f'{object_depths[i]:.2f}m'
                    label += f' ({depth_text})'

                # วาดกรอบ Bounding Box และหมายเลขวัตถุ
                plot_one_box((x1, y1, x2, y2), frame, label=label, color=(0, 0, 255), line_thickness=2)

                # วาดหมายเลขวัตถุที่มุม Bounding Box
                cv2.putText(frame, str(obj_id), (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # แสดงผลลัพธ์
        cv2.imshow("Front View - YOLOP", frame)

        # ----------------- พิมพ์ข้อมูลการตรวจจับ -----------------
        print(f"✅ Detection Results - Objects: {len(det_pred[0]) if det_pred[0] is not None else 0}")
        for i, depth in enumerate(object_depths):
            print(f"   ▶ Object {i+1}: {depth:.2f} m" if depth is not None else f"   ▶ Object {i+1}: No Depth Data")

        # รอให้กดปุ่ม 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()
