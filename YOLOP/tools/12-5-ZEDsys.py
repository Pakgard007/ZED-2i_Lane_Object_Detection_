import os, sys
import numpy as np
import cv2
import pyzed.sl as sl
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
# Import YOLOP utilities
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression
sys.path.append('/home/mag/zed/samples/object detection/birds eye viewer/python')
import ogl_viewer.viewer as gl
import platform

# ตรวจสอบว่าเป็น Jetson หรือไม่
is_jetson = False
if platform.uname().machine.startswith('aarch64'):
    is_jetson = True

# กำหนดพารามิเตอร์ YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ใช้ GPU ถ้ามี
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_net(cfg)
weights_path = "/home/mag/satoi/python/YOLOP/weights/End-to-end.pth"
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

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

# ฟังก์ชันตรวจจับ Drivable Area จาก YOLOP
def detect_drivable_area(image, model):
    input_image = transform(image).to(device)
    input_image = input_image.unsqueeze(0)

    with torch.no_grad():
        _, da_seg_out, _ = model(input_image)

    da_seg_out = F.interpolate(da_seg_out, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
    da_seg_mask = torch.max(da_seg_out, 1)[1].squeeze().cpu().numpy()

    return da_seg_mask

# ฟังก์ชันตรวจจับวัตถุ
def detect_objects(zed, runtime_parameters):
    objects = sl.Objects()
    zed.retrieve_objects(objects, runtime_parameters)
    return objects

# ฟังก์ชันคำนวณเส้นทางใหม่
def calculate_new_path(drivable_area_mask, object_detected):
    if object_detected:
        print("🚫 วัตถุขวางทาง! กำลังคำนวณเส้นทางใหม่...")
        return "กำหนดเส้นทางใหม่"
    else:
        return "เดินทางได้ตามปกติ"

# ฟังก์ชันแสดงผล OpenGL
def display_openGL(zed, viewer, objects, pc_resolution, point_cloud):
    while True:
        # ดึงข้อมูลจาก ZED
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # แสดงผล Point Cloud ด้วย OpenGL
            viewer.updateData(point_cloud, objects)
            cv2.imshow("ZED 3D View", global_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

# เริ่มต้นการทำงาน
def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ไม่สามารถเปิดกล้องได้!")
        exit()

    # สร้าง Viewer สำหรับ OpenGL
    viewer = gl.GLViewer()
    pc_resolution = sl.Resolution(1280, 720)
    viewer.init(zed.get_camera_information().camera_model, pc_resolution, True)
    point_cloud = sl.Mat(pc_resolution.width, pc_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # เริ่มต้นการตรวจจับวัตถุ
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50

    while True:
        # ดึงข้อมูลจาก ZED
        image = sl.Mat()
        zed.grab()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()
        # แปลงจาก RGBA เป็น RGB ก่อน
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # ตรวจจับ Drivable Area
        da_seg_mask = detect_drivable_area(frame_rgb, model)

        # ตรวจจับวัตถุจาก ZED SDK
        objects = detect_objects(zed, runtime_parameters)

        # ตรวจสอบการพบวัตถุขวางทาง
        object_detected = False
        for obj in objects.object_list:
            if obj.label == sl.OBJECT_CLASS.PERSON:
                object_detected = True
                break

        # คำนวณเส้นทางใหม่
        path_status = calculate_new_path(da_seg_mask, object_detected)
        print(f"เส้นทาง: {path_status}")

        # แสดงผล OpenGL 3D
        display_openGL(zed, viewer, objects, pc_resolution, point_cloud)

# ปิดกล้อง ZED
zed.close()
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
