import argparse
import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from tqdm import tqdm

# 📌 **Normalize image for YOLOP model**
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# 📌 **Perspective Transformation for BEV**
def perspective_transform(image):
    height, width = image.shape[:2]
    
    # **จุดที่ใช้แปลงมุมมอง**
    tl = (650, 800)  # Top-left (ซ้ายบน) → ขอบซ้ายของถนน
    bl = (450, 1000)  # Bottom-left (ซ้ายล่าง) → ขอบซ้ายใกล้จอ
    tr = (1050, 800)  # Top-right (ขวาบน) → ขอบขวาของถนน
    br = (1250, 1000)  # Bottom-right (ขวาล่าง) → ขอบขวาใกล้จอ

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(image, matrix, (640, 480))
    return transformed_frame, pts1

# 📌 **Function สำหรับประมวลผล YOLOP**
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
        det_out, da_seg_out, _ = model(img)  # ❌ ตัด Lane Line ออก
        t2 = time_synchronized()

        _, _, height, width = img.shape
        pad_w, pad_h = map(int, shapes[1][1])
        ratio = shapes[1][0][1]

        # 📌 **Process Drivable Area Mask**
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8) * 255

        # 📌 **Convert to Colored Image**
        da_seg_mask_colored = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
        da_seg_mask_colored[da_seg_mask > 0] = [0, 255, 0]  

        # 📌 **Apply BEV Transformation**
        bev_view, pts1 = perspective_transform(da_seg_mask_colored)

        # 📌 **วาดจุด 4 จุดที่ใช้แปลงเป็นมุม BEV บนภาพ Front View**
        for pt in pts1:
            cv2.circle(img_det, tuple(pt.astype(int)), 8, (0, 0, 255), -1)

        # 📌 **แสดงผล**
        front_view_resized = cv2.resize(img_det, (640, 720))
        bev_view_resized = cv2.resize(bev_view, (640, 720))

        # ✅ **แสดงภาพ Front View + BEV View**
        combined_view = np.hstack((front_view_resized, bev_view_resized))

        combined_save_path = os.path.join(opt.save_dir, f"combined_{i}.jpg")
        cv2.imwrite(combined_save_path, combined_view)

        cv2.imshow("Combined View", combined_view)
        cv2.waitKey(1)

        print(f'Results saved to {os.path.join(opt.save_dir)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOP/weights/End-to-end.pth', help='model.pth path')
    parser.add_argument('--source', type=str, default='/home/satoi/final.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/satoi/python/result', help='save results directory')
    opt = parser.parse_args()
    
    with torch.no_grad():
        detect(cfg, opt)
