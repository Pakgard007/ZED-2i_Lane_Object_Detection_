import argparse
import os, sys
import shutil
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from tqdm import tqdm

# Normalize ค่า input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def get_bird_eye_view(image, src_points, dst_points, output_size):
    """
    แปลงภาพเป็น Bird's Eye View (BEV)
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bird_eye_view = cv2.warpPerspective(image, M, output_size)

    return bird_eye_view

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
        det_out, da_seg_out, _ = model(img)  # ใช้แค่ Drivable Area เท่านั้น
        t2 = time_synchronized()

        _, _, height, width = img.shape
        pad_w, pad_h = map(int, shapes[1][1])
        ratio = shapes[1][0][1]

        # ✅ ดึงข้อมูลเฉพาะ Drivable Area
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8) * 255

        # ✅ กำหนดจุด Perspective Transform ให้ภาพ BEV อยู่ตรงกลาง
        src_points = np.float32([
            [350, 720],   # ซ้ายล่าง
            [930, 720],   # ขวาล่าง
            [500, 450],   # ซ้ายบน (ยกสูงขึ้น)
            [780, 450]    # ขวาบน (ยกสูงขึ้น)
        ])

        dst_points = np.float32([
            [400, 1080],  # ซ้ายล่าง
            [800, 1080],  # ขวาล่าง
            [500, 0],     # ซ้ายบน
            [700, 0]      # ขวาบน
        ])

        output_size = (1550, 1080)  # ✅ ปรับขนาดให้เต็มเฟรม

        drivable_bev = get_bird_eye_view(da_seg_mask, src_points, dst_points, output_size)

        # ✅ แสดงพื้นที่ขับขี่เป็นสีเขียว
        drivable_bev_colored = cv2.cvtColor(drivable_bev, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(drivable_bev_colored)
        overlay[drivable_bev == 255] = [0, 255, 0]  # Green for drivable area

        combined_bev = cv2.addWeighted(drivable_bev_colored, 1, overlay, 0.7, 0)

        # ✅ แสดงผลแนวตั้ง
        bird_eye_view_resized = cv2.resize(combined_bev, (960, 720))
        front_view_resized = cv2.resize(img_det, (480, 720))

        combined_view = np.hstack((front_view_resized, bird_eye_view_resized))

        combined_save_path = os.path.join(opt.save_dir, f"combined_{i}.jpg")
        cv2.imwrite(combined_save_path, combined_view)

        cv2.imshow("Combined View", combined_view)
        cv2.waitKey(1)
        print(f'Results saved to {Path(opt.save_dir)}')

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
