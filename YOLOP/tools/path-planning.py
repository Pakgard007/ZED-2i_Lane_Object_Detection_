import argparse
import os, sys
import shutil
import time
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from simple_pid import PID

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.core.function import AverageMeter

# Normalize for YOLOP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# PID Controller Parameters
pid = PID(Kp=0.5, Ki=0.01, Kd=0.1, setpoint=0)

# Minimum distance from lane in pixels (1 meter equivalent)
MIN_DISTANCE_FROM_LANE = 50


def detect(cfg, opt):
    """Runs YOLOP inference and applies PID control for path planning."""
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")
    
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)
    half = device.type != 'cpu'

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    dataset = LoadImages(opt.source, img_size=opt.img_size)
    t0 = time.time()

    for i, (path, img, img_det, vid_cap, shapes) in enumerate(dataset):
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        img = img.unsqueeze(0) if img.ndimension() == 3 else img

        # Inference
        det_out, da_seg_out, ll_seg_out = model(img)  # Include Lane Line Segmentation
        det_pred = non_max_suppression(det_out[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
        det = det_pred[0]

        # Extract Drivable Area Mask
        da_predict = da_seg_out[:, :, :, :]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=1, mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (img_det.shape[1], img_det.shape[0]))  # Resize mask
        
        # Extract Lane Line Mask
        ll_predict = ll_seg_out[:, :, :, :]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=1, mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (img_det.shape[1], img_det.shape[0]))  # Resize mask
        
        # Process PID based on center of drivable area
        contours, _ = cv2.findContours(da_seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Ensure safe distance from lane lines
                lane_pixels = np.where(ll_seg_mask[cy, :] > 0)[0]
                if lane_pixels.size > 0:
                    left_lane = np.min(lane_pixels)
                    right_lane = np.max(lane_pixels)
                    if cx - left_lane < MIN_DISTANCE_FROM_LANE:
                        cx = left_lane + MIN_DISTANCE_FROM_LANE
                    if right_lane - cx < MIN_DISTANCE_FROM_LANE:
                        cx = right_lane - MIN_DISTANCE_FROM_LANE
                
                # Check for obstacles and draw bounding boxes
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2
                    if y1 < cy < y2:  # Object is in front
                        cx = cx - 50 if obj_center_x > cx else cx + 50  # Adjust path left or right
                        cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box
                
                error = cx - (img_det.shape[1] // 2)
                control_signal = pid(error)
                print(f"PID Control: {control_signal}")
                
                # Draw center point and lane boundary on the image
                cv2.circle(img_det, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(img_det, (cx, cy), (cx, cy - 30), (0, 255, 255), 2)  # Visual path line
                
        # Display Results
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        cv2.imshow('Drivable Area', img_det)
        cv2.waitKey(1)

    print('Results saved to', opt.save_dir)
    print('Done in', time.time() - t0, 'seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/mag/satoi/python/YOLOP/weights/End-to-end.pth', help='Path to trained model')
    parser.add_argument('--source', type=str, default='/home/mag/satoi/final.mp4', help='Path to video/image source')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', default='cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(cfg, opt)
