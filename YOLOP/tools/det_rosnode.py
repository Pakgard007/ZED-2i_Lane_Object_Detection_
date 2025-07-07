import os
import sys
import math
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Import YOLOP utilities
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression
from lib.utils import plot_one_box

# Normalize for YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

class ObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detector_node')

        # ROS Publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.model = self.load_yolop_model("/home/mag/satoi/python/YOLOP/weights/End-to-end.pth", self.device)

        # Init ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Cannot open ZED camera")
            sys.exit(1)

        self.image = sl.Mat()
        self.depth_map = sl.Mat()

        # Load calibration data
        calib_data = np.load("camera_calibration.npz")
        self.mtxL = calib_data["camera_matrix_L"]
        self.distL = np.array([
            -0.07709319689152208,
             0.06756180189133752,
             0.00015006759935512075,
            -6.006342505065124e-05,
            -0.028545020615709165
        ])

    def load_yolop_model(self, weights_path, device):
        model = get_net(cfg)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        self.get_logger().info(f"Model loaded: {weights_path}")
        return model

    def create_roi(self, image, roi_width=150, roi_height=500):
        h, w = image.shape[:2]
        x_start = (w - roi_width) // 2
        y_start = (h - roi_height) // 2
        cv2.rectangle(image, (x_start, y_start),
                      (x_start + roi_width, y_start + roi_height),
                      (255, 255, 255), 2)
        roi = image[y_start:y_start + roi_height, x_start:x_start + roi_width]
        return roi, (x_start, y_start, roi_width, roi_height)

    def get_depth_for_detections(self, depth_map, detections, roi_coords):
        object_depths = []
        if detections is None or len(detections) == 0:
            return object_depths
        x_roi, y_roi, roi_w, roi_h = roi_coords
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            if x1 < x_roi or x2 > (x_roi + roi_w) or y1 < y_roi or y2 > (y_roi + roi_h):
                object_depths.append(None)
                continue
            obj_depth_roi = depth_map[y1:y2, x1:x2]
            valid_depth = obj_depth_roi[np.isfinite(obj_depth_roi)]
            if len(valid_depth) == 0:
                object_depths.append(None)
            else:
                object_depths.append(np.median(valid_depth))
        return object_depths

    def detect_obstacles(self, image):
        input_image = transform(image).to(self.device)
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.model(input_image)
        _, _, height, width = input_image.shape
        da_seg_out = F.interpolate(da_seg_out, size=(height, width), mode='bilinear', align_corners=False)
        ll_seg_out = F.interpolate(ll_seg_out, size=(height, width), mode='bilinear', align_corners=False)
        da_seg_mask = torch.max(da_seg_out, 1)[1].squeeze().cpu().numpy()
        ll_seg_mask = torch.max(ll_seg_out, 1)[1].squeeze().cpu().numpy()
        det_pred = non_max_suppression(det_out[0], conf_thres=0.3, iou_thres=0.5)
        return da_seg_mask, ll_seg_mask, det_pred

    def decision_making(self, object_depths):
        if not object_depths:
            return "SAFE"
        min_depth = min([d for d in object_depths if d is not None], default=None)
        if min_depth is None:
            return "SAFE"
        elif min_depth < 3.0:
            return "STOP"
        elif min_depth < 5.0:
            return "SLOW"
        else:
            return "SAFE"

    def overlay_segmentation(self, image, mask, color=(0, 255, 0), alpha=0.4):
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[mask > 0] = color
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX)
        return blended

    def run(self):
        self.get_logger().info("Starting main loop. Press 'q' to quit.")
        while rclpy.ok():
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                frame = self.image.get_data()[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                undistorted_frame = cv2.undistort(frame, self.mtxL, self.distL)

                da_seg_mask, ll_seg_mask, det_pred = self.detect_obstacles(undistorted_frame)

                # Overlay lane and drivable area on front view
                overlay_da = self.overlay_segmentation(undistorted_frame.copy(), da_seg_mask, (0, 255, 0))
                overlay_ll = self.overlay_segmentation(overlay_da, ll_seg_mask, (0, 0, 255))

                # Create ROI
                frame_with_roi = overlay_ll.copy()
                roi, roi_coords = self.create_roi(frame_with_roi)

                # Get depth of detected objects inside ROI
                object_depths = self.get_depth_for_detections(self.depth_map.get_data(), det_pred[0], roi_coords)

                # Draw bounding boxes with depth
                if det_pred[0] is not None:
                    for i, det in enumerate(det_pred[0]):
                        if len(det) < 6:
                            continue
                        x1, y1, x2, y2, conf, cls = det[:6]
                        x_roi, y_roi, roi_w, roi_h = roi_coords
                        if x1 < x_roi or x2 > (x_roi + roi_w) or y1 < y_roi or y2 > (y_roi + roi_h):
                            continue
                        label = f"Obj {i+1}"
                        if object_depths[i] is not None:
                            depth_text = f"{object_depths[i]:.2f} m"
                            label += f" ({depth_text})"
                        else:
                            depth_text = "No Depth"
                        plot_one_box((x1, y1, x2, y2), frame_with_roi, label=label,
                                     color=(0,0,255), line_thickness=2)
                        cv2.putText(frame_with_roi, str(i+1), (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                # Make decision
                decision = self.decision_making(object_depths)

                # Prepare and publish cmd_vel message
                twist = Twist()
                if decision == "STOP":
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                elif decision == "SLOW":
                    twist.linear.x = 0.1
                    twist.angular.z = 0.0
                else:
                    twist.linear.x = 0.4
                    twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)

                self.get_logger().info(f"Decision: {decision} - Published cmd_vel: linear.x={twist.linear.x}")

                # Show result
                cv2.imshow("Front View - Detection + ROI", frame_with_roi)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.zed.close()    
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
