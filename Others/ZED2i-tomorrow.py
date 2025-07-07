import argparse
import os
import sys

import numpy as np
import cv2
import torch
import pyzed.sl as sl
from pathlib import Path
from numpy import random
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_one_box


def load_yolov5_model(weights_path, device):
    """Load the YOLOv5 model."""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model = model.to(device)
    return model

def detect_objects_yolov5(model, img, device, conf_thres=0.25, iou_thres=0.45):
    """Run YOLOv5 detection on the given image."""
    img = img / 255.0  # Normalize image
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    img = img.float()
    detections = model(img, size=640)[0]
    detections = non_max_suppression(detections, conf_thres, iou_thres)
    return detections

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to YOLOv5 weights')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='Device to use for inference: cpu or cuda')
    args = parser.parse_args()

    # Load YOLOv5 model
    device = select_device(args.device)
    model = load_yolov5_model(args.weights, device)

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)

    runtime_parameters = sl.RuntimeParameters()

    # Start real-time detection
    print("Starting YOLOv5 real-time detection with ZED...")
    image = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img_np = image.get_data()[:, :, :3]  # Extract RGB image

            # Resize image for YOLOv5
            img_resized = cv2.resize(img_np, (640, 640))

            # Detect objects
            detections = detect_objects_yolov5(model, img_resized, device, args.conf_thres, args.iou_thres)

            # Draw bounding boxes
            if detections[0] is not None:
                for det in detections[0]:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    plot_one_box([x1, y1, x2, y2], img_np, label=label, color=[random.randint(0, 255) for _ in range(3)], line_thickness=2)

            # Display the results
            cv2.imshow("YOLOv5 Detection with ZED", img_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
