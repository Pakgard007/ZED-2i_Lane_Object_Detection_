import argparse
import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pathlib import Path
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from tqdm import tqdm

# Normalize image for YOLOP model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

def extract_contours(da_seg_mask):
    """
    ใช้ Contour Tracking เพื่อตรวจจับขอบเขตของ Drivable Area
    """
    _, binary_mask = cv2.threshold(da_seg_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def determine_direction(da_seg_mask):
    """
    ใช้ Contour Tracking และ Centroid เพื่อกำหนดทิศทางการเคลื่อนที่ของรถ
    """
    contours = extract_contours(da_seg_mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  
            cy = int(M["m01"] / M["m00"])  

            frame_width = da_seg_mask.shape[1]

            if cx < frame_width // 3:
                return "Turn Left", (cx, cy)
            elif cx > 2 * frame_width // 3:
                return "Turn Right", (cx, cy)
            else:
                return "Go Forward", (cx, cy)
    
    return "No Data", (0, 0)

def detect(cfg, opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger, opt.device)
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)
    half = device.type != 'cpu'

    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()

    dataset = LoadImages(opt.source, img_size=opt.img_size) if not opt.source.isnumeric() else LoadStreams(opt.source, img_size=opt.img_size)

    for i, (path, img, img_det, _, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        det_out, da_seg_out, _ = model(img)  
        t2 = time_synchronized()

        _, _, height, width = img.shape
        pad_w, pad_h = map(int, shapes[1][1])
        ratio = shapes[1][0][1]

        # Process drivable area mask
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8) * 255

        # Convert to colored image
        da_seg_mask_colored = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
        da_seg_mask_colored[da_seg_mask > 0] = [0, 255, 0]  

        # 🔹 **ใช้ Contour Tracking เพื่อติดตามขอบเขตของ Drivable Area**
        contours = extract_contours(da_seg_mask)
        cv2.drawContours(da_seg_mask_colored, contours, -1, (255, 0, 0), 2)  

        # 🔹 **คำนวณการตัดสินใจควบคุมรถ**
        direction, centroid = determine_direction(da_seg_mask)

        # วาดจุด Centroid และแสดงข้อความทิศทาง
        if centroid != (0, 0):
            cv2.circle(da_seg_mask_colored, centroid, 10, (0, 0, 255), -1)
            cv2.putText(da_seg_mask_colored, direction, (centroid[0] - 50, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # Resize for visualization
        front_view_resized = cv2.resize(img_det, (640, 720))
        drivable_area_resized = cv2.resize(da_seg_mask_colored, (640, 720))

        # ✅ **แสดงเฉพาะ Drivable Area และ Front View**
        combined_view = np.hstack((front_view_resized, drivable_area_resized))

        combined_save_path = os.path.join(opt.save_dir, f"combined_{i}.jpg")
        cv2.imwrite(combined_save_path, combined_view)

        # Show results
        cv2.imshow("Combined View", combined_view)
        cv2.waitKey(1)

        print(f'Results saved to {Path(opt.save_dir)} | Decision: {direction}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOP/weights/End-to-end.pth', help='model.pth path')
    parser.add_argument('--source', type=str, default='/home/mag/satoi/final.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/mag/satoi/python/result', help='save results directory')
    opt = parser.parse_args()
    
    with torch.no_grad():
        detect(cfg, opt)
