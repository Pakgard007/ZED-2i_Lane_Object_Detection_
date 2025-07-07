########################################################################
# Copyright (c) 2022, STEREOLABS.
# All rights reserved.
########################################################################

import sys
import numpy as np
import cv2
import pyzed.sl as sl
import argparse
import platform
from collections import deque

sys.path.append('/home/mag/zed/samples/object detection/birds eye viewer/python')
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

# เช็คว่าเป็น Jetson หรือไม่
is_jetson = platform.uname().machine.startswith('aarch64')

def main(opt):
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_maximum_distance = 10.0

    # Apply arguments
    parse_args(opt, init_params)

    is_playback = len(opt.input_svo_file) > 0
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

    # GUI Configuration
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

        viewer = gl.GLViewer()
        pc_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio)
        viewer.init(zed.get_camera_information().camera_model, pc_resolution, obj_param.enable_tracking)
        point_cloud = sl.Mat(pc_resolution.width, pc_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        image_left = sl.Mat()
        cam_w_pose = sl.Pose()
        image_scale = (display_resolution.width / camera_config.resolution.width, display_resolution.height / camera_config.resolution.height)

    objects = sl.Objects()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    window_name = "ZED | 3D Object Tracking"
    gl_viewer_available = True

    printHelp()
    while True:
        if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool:
            break

        if zed.retrieve_objects(objects, detection_parameters_rt) == sl.ERROR_CODE.SUCCESS:
            if not opt.disable_gui:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_render_left = image_left.get_data()
                np.copyto(image_left_ocv, image_render_left)

                track_view_generator.generate_view(objects, image_left_ocv, image_scale, cam_w_pose, image_track_ocv, objects.is_tracked)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                viewer.updateData(point_cloud, objects)
                gl_viewer_available = viewer.is_available()

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
        viewer.exit()
        point_cloud.free()
        image_left.free()

    zed.disable_object_detection()
    zed.close()

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

    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA

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
