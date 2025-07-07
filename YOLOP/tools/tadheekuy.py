import os, sys
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

# เปิดกล้อง ZED 2i
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  
init_params.camera_fps = 60
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  
init_params.coordinate_units = sl.UNIT.METER  

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image = sl.Mat()
depth_map = sl.Mat()

# โหลดโมเดล YOLOP
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

# สร้าง ROI ที่แน่ใจว่าอยู่ตรงกลางของภาพ
def create_roi(image, roi_width_pixels=300, roi_height_pixels=700):
    height, width = image.shape[:2]
    x_start = (width - roi_width_pixels) // 2
    y_start = (height - roi_height_pixels) // 2

    roi = image[y_start:y_start + roi_height_pixels, x_start:x_start + roi_width_pixels]
    
    # **วาดกรอบ ROI ในภาพ**
    # แนะนำ: วาดลงใน frame_with_ll ซึ่งจะแสดงจริง
    cv2.rectangle(frame_with_ll, (x_start, y_start), (x_start + roi_width_pixels, y_start + roi_height_pixels), (255, 255, 255), 2)

    return roi, (x_start, y_start, roi_width_pixels, roi_height_pixels)

# ฟังก์ชันดึง Depth ของวัตถุภายใน ROI เท่านั้น
def get_depth_for_detections(depth_map, detections, roi_coords):
    object_depths = []
    
    if detections is None or len(detections) == 0:
        return object_depths  

    x_roi, y_roi, roi_width, roi_height = roi_coords

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  

        # เช็ควัตถุอยู่ใน ROI หรือไม่
        if x1 < x_roi or x2 > (x_roi + roi_width) or y1 < y_roi or y2 > (y_roi + roi_height):
            object_depths.append(None)
            continue  

        obj_depth_roi = depth_map[y1:y2, x1:x2]
        valid_depth = obj_depth_roi[np.isfinite(obj_depth_roi)]

        if len(valid_depth) == 0:
            object_depths.append(None)
        else:
            object_depths.append(np.median(valid_depth))
    
    return object_depths

# ตรวจจับ Drivable Area และ Lane
def detect_obstacles(image, model):
    input_image = transform(image).to(device).unsqueeze(0)

    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(input_image)

    _, _, height, width = input_image.shape
    da_seg_out = F.interpolate(da_seg_out, size=(height, width), mode='bilinear', align_corners=False)
    ll_seg_out = F.interpolate(ll_seg_out, size=(height, width), mode='bilinear', align_corners=False)

    da_seg_mask = torch.max(da_seg_out, 1)[1].squeeze().cpu().numpy()
    ll_seg_mask = torch.max(ll_seg_out, 1)[1].squeeze().cpu().numpy()
    det_pred = non_max_suppression(det_out[0], conf_thres=0.25, iou_thres=0.45)

    return da_seg_mask, ll_seg_mask, det_pred

# ตัดสินใจเคลื่อนที่ของรถ 🚗
def decision_making(object_depths):
    if not object_depths:
        return "✅ SAFE: Continue Moving"

    min_depth = min([d for d in object_depths if d is not None], default=None)

    if min_depth is None:
        return "✅ SAFE: Continue Moving"
    elif min_depth < 3.0:
        return "🚫 STOP: Obstacle Ahead!"
    elif min_depth < 5.0:
        return "⚠ SLOW DOWN: Obstacle Ahead"
    else:
        return "✅ SAFE: Continue Moving"

# Overlay Drivable Area และ Lane
def overlay_segmentation(image, mask, color=(0, 255, 0), alpha=0.4):
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask > 0] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/mag/satoi/seminar-pre/result-obj3.mp4', fourcc, 30.0, (1280, 720))

# อ่านภาพจากกล้อง ZED 2i และตรวจจับ
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        frame = cv2.cvtColor(image.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)  

        da_seg_mask, ll_seg_mask, det_pred = detect_obstacles(frame, model)

        frame_with_da = overlay_segmentation(frame.copy(), da_seg_mask, (0, 255, 0))  
        frame_with_ll = overlay_segmentation(frame_with_da, ll_seg_mask, (0, 0, 255))  
        roi, roi_coords = create_roi(frame_with_ll)
        object_depths = get_depth_for_detections(depth_map.get_data(), det_pred[0], roi_coords)

        # แสดงผลลัพธ์ใน Terminal
        if object_depths:
            for i, depth in enumerate(object_depths):
                if depth is not None:
                    print(f"▶ Object {i+1}: {depth:.2f} m")
                else:
                    print(f"▶ Object {i+1}: No Depth Data")

        # วาด Bounding Box และหมายเลขวัตถุ + ระยะห่าง
        if det_pred[0] is not None:
            for i, det in enumerate(det_pred[0]):
                if len(det) < 6:
                    continue  

                x1, y1, x2, y2, conf, cls = det[:6]

                # ✅ เงื่อนไขกรอบต้องอยู่ใน ROI
                x_roi, y_roi, roi_width, roi_height = roi_coords
                if x1 < x_roi or x2 > (x_roi + roi_width) or y1 < y_roi or y2 > (y_roi + roi_height):
                    continue  # ข้ามวัตถุที่อยู่นอก ROI

                obj_id = i + 1
                label = f'Object {obj_id}'

                if object_depths[i] is not None:
                    depth_text = f'{object_depths[i]:.2f} m'
                    label += f' ({depth_text})'
                else:
                    depth_text = "No Depth Data"

                plot_one_box((x1, y1, x2, y2), frame_with_ll, label=label, color=(0, 0, 255), line_thickness=2)
                cv2.putText(frame_with_ll, str(obj_id), (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                print(f"▶ Object {obj_id}: {depth_text}")


        decision = decision_making(object_depths)
        print(f"🛑 Driving Decision: {decision}")
            

        cv2.imshow("Front View - YOLOP", frame_with_ll)
        out.write(frame_with_ll)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
zed.close()
cv2.destroyAllWindows()
