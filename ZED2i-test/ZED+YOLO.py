import sys
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pyzed.sl as sl
import argparse
import platform
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from shapely.geometry import Polygon, box


sys.path.append('/home/mag/zed/samples/object detection/birds eye viewer/python')
sys.path.append('/home/mag/satoi/python/YOLOP')  # Path to YOLOP root

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

is_jetson = platform.uname().machine.startswith('aarch64')

class DriveStatusPublisher(Node):
    def __init__(self):
        super().__init__('drive_status_publisher')
        self.publisher_ = self.create_publisher(String, '/drive_status', 10)

    def publish_status(self, status_str):
        msg = String()
        msg.data = status_str
        self.publisher_.publish(msg)

def detect_yolop(model, device, frame):
    input_image = transform(frame).to(device).unsqueeze(0)
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(input_image)
    _, _, height, width = input_image.shape
    da_seg_out = F.interpolate(da_seg_out, size=(height, width), mode='bilinear', align_corners=False)
    ll_seg_out = F.interpolate(ll_seg_out, size=(height, width), mode='bilinear', align_corners=False)

    da_seg_mask = torch.max(da_seg_out, 1)[1].squeeze().cpu().numpy()
    ll_seg_mask = torch.max(ll_seg_out, 1)[1].squeeze().cpu().numpy()
    return da_seg_mask, ll_seg_mask

def load_yolop_model(weights_path, device):
    model = get_net(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model

def main(opt):
    rclpy.init()
    drive_status_node = DriveStatusPublisher()

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_maximum_distance = 10.0

    parse_args(opt, init_params)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open Error:", status)
        exit()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    batch_parameters = sl.BatchParameters()
    if opt.enable_batching_reid:
        batch_parameters.enable = True
        batch_parameters.latency = 3.0

    obj_param = sl.ObjectDetectionParameters(batch_trajectories_parameters=batch_parameters)
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST if is_jetson else sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    obj_param.enable_tracking = True

    if zed.enable_object_detection(obj_param) != sl.ERROR_CODE.SUCCESS:
        print("enable_object_detection() failed")
        zed.close()
        exit()

    detection_confidence = 60
    detection_parameters_rt = sl.ObjectDetectionRuntimeParameters(detection_confidence)
    detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE, sl.OBJECT_CLASS.PERSON]
    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.PERSON] = detection_confidence
    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.VEHICLE] = detection_confidence

    camera_configuration = zed.get_camera_information().camera_configuration

    quit_bool = False
    if not opt.disable_gui:
        image_aspect_ratio = camera_configuration.resolution.width / camera_configuration.resolution.height
        requested_low_res_w = min(1280, camera_configuration.resolution.width)
        display_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio)

        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
        camera_config = camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)

        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance*1000, batch_parameters.latency)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)

        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        global_image = np.full((display_resolution.height, display_resolution.width + tracks_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        image_left = sl.Mat()
        cam_w_pose = sl.Pose()
        image_scale = (display_resolution.width / camera_config.resolution.width, display_resolution.height / camera_config.resolution.height)

    objects = sl.Objects()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    window_name = "ZED | 3D Object Tracking"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")
    yolop_model = load_yolop_model("/home/mag/satoi/python/YOLOP/weights/End-to-end.pth", device)

    printHelp()
    while True:
        if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool:
            break

        if zed.retrieve_objects(objects, detection_parameters_rt) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            frame_rgba = image_left.get_data()
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

            da_seg_mask, ll_seg_mask = detect_yolop(yolop_model, device, frame_bgr)
            
            da_overlay = np.zeros_like(frame_bgr)
            da_overlay[da_seg_mask > 0] = (0, 255, 0)  # Drivable Area - Green

            ll_overlay = np.zeros_like(frame_bgr)
            ll_overlay[ll_seg_mask > 0] = (0, 0, 255)  # Lane Line - Red

            combined_overlay = cv2.addWeighted(da_overlay, 0.8, ll_overlay, 0.8, 0)
            blended = cv2.addWeighted(frame_bgr, 0.6, combined_overlay, 0.4, 0)

            # ต้องใช้ display_resolution เพื่อให้สอดคล้องกับภาพที่แสดงผล
            
            obstacle_in_path = False

            for obj in objects.object_list:
                if obj.label in [sl.OBJECT_CLASS.PERSON, sl.OBJECT_CLASS.VEHICLE]:
                    z = abs(obj.position[2])
                    if z < 5.0:
                        print(f"🟥 BRAKE: Object ID={obj.id}, dist={z:.2f}m")
                        drive_status_node.publish_status("brake")
                        obstacle_in_path = True
                        break
                    elif 5.0 <= z < 8.0:
                        print(f"🟧 SLOW: Object ID={obj.id}, dist={z:.2f}m")
                        drive_status_node.publish_status("slow")
                        obstacle_in_path = True
                        break
                    else:
                        print(f"🟩 GO: Object ID={obj.id}, dist={z:.2f}m")

            if not obstacle_in_path:
                print("✅ GO: Path is clear")
                drive_status_node.publish_status("go")


            np.copyto(image_left_ocv, cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA))
            track_view_generator.generate_view(objects, image_left_ocv, image_scale, cam_w_pose, image_track_ocv, objects.is_tracked)
            # อยู่ใต้ track_view_generator.generate_view(...)
            center_x = tracks_resolution.width // 2
            camera_bottom_y = tracks_resolution.height - 50  # ปรับเลขนี้ให้เส้นหยุดพอดีกับหัวกล้อง
            cv2.line(image_track_ocv, (center_x, 0), (center_x, camera_bottom_y), (0, 0, 255), 2)

            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])

            cv2.imshow(window_name, global_image)
            key = cv2.waitKey(10)
            if key == ord('q'):
                quit_bool = True
            elif key == ord('i'):
                track_view_generator.zoomIn()
            elif key == ord('o'):
                track_view_generator.zoomOut()
            elif key == ord('p'):
                detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.PERSON]
                print("Filter: Person only")
            elif key == ord('v'):
                detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE]
                print("Filter: Vehicle only")
            elif key == ord('c'):
                detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE, sl.OBJECT_CLASS.PERSON]
                print("Filter: Both vehicle and person")

    if not opt.disable_gui:
        image_left.free()

    zed.disable_object_detection()
    zed.close()
    rclpy.shutdown()

def parse_args(opt, init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("Using SVO File:", opt.input_svo_file)
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit():
            parts = ip_str.split(':')
            if len(parts) == 2:
                init.set_from_stream(parts[0], int(parts[1]))
            else:
                init.set_from_stream(ip_str)
            print("Using Stream input:", ip_str)

    res_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
    }
    if opt.resolution in res_map:
        init.camera_resolution = res_map[opt.resolution]

def printHelp():
    print("\nHotkeys:")
    print(" i : Zoom in tracking view")
    print(" o : Zoom out tracking view")
    print(" p : Filter only Person")
    print(" v : Filter only Vehicle")
    print(" c : Show both Person & Vehicle")
    print(" q : Quit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='', help='Path to an .svo file')
    parser.add_argument('--ip_address', type=str, default='', help='Stream IP address (e.g. 192.168.1.1:30000)')
    parser.add_argument('--resolution', type=str, default='HD1080', help='Camera resolution: HD2K, HD1080, HD720, etc.')
    parser.add_argument('--disable_gui', action='store_true', help='Disable GUI for Jetson/low-end devices')
    parser.add_argument('--enable_batching_reid', action='store_true', help='Enable re-identification for object tracking')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Error: Use either --input_svo_file or --ip_address, not both.")
        exit()

    main(opt)