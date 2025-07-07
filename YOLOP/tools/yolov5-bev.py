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
from pathlib import Path
from tqdm import tqdm

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams

# Normalize image for YOLOP model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

def transform_to_bev(frame, da_seg_mask, ll_seg_mask):
    """
    แปลงมุมมองภาพจาก Front View เป็น Bird's Eye View (BEV)
    """
    h, w = frame.shape[:2]

    # กำหนด 4 จุดของภาพต้นฉบับ (เลนถนน)
    src_points = np.float32([
        (200, h // 2),  # Top-left
        (w - 200, h // 2),  # Top-right
        (50, h),  # Bottom-left
        (w - 50, h)  # Bottom-right
    ])

    # กำหนดจุดปลายทางในมุมมอง BEV
    dst_points = np.float32([
        (200, 0),  # Top-left
        (w - 200, 0),  # Top-right
        (200, h),  # Bottom-left
        (w - 200, h)  # Bottom-right
    ])

    # คำนวณ Perspective Transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_frame = cv2.warpPerspective(frame, matrix, (w, h))
    bev_da_mask = cv2.warpPerspective(da_seg_mask, matrix, (w, h))
    bev_ll_mask = cv2.warpPerspective(ll_seg_mask, matrix, (w, h))

    # รวม Lane Line และ Drivable Area ในมุมมอง BEV
    bev_combined = cv2.addWeighted(bev_frame, 1, bev_da_mask, 1, 0)
    bev_combined = cv2.addWeighted(bev_combined, 1, bev_ll_mask, 1, 0)

    return bev_combined

def detect(cfg, opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')

    # เลือกอุปกรณ์ (CPU หรือ CUDA)
    device = select_device(logger, opt.device)
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)
    half = device.type != 'cpu'

    # โหลดโมเดล YOLOP และรองรับการใช้ CPU
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location='cpu')  # บังคับให้ใช้ CPU ถ้าไม่มี CUDA
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()

    # โหลดวิดีโอ
    dataset = LoadImages(opt.source, img_size=opt.img_size) if not opt.source.isnumeric() else LoadStreams(opt.source, img_size=opt.img_size)
    vid_writer = None

    for i, (path, img, img_det, _, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = model(img)  # เพิ่ม Lane Line Segmentation
        t2 = time_synchronized()

        _, _, height, width = img.shape
        pad_w, pad_h = map(int, shapes[1][1])
        ratio = shapes[1][0][1]

        # Process drivable area mask
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8) * 255

        # Process lane line mask
        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8) * 255

        # แปลงเป็นภาพสี
        da_seg_mask_colored = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
        da_seg_mask_colored[da_seg_mask > 0] = [0, 255, 0]  # Green for drivable area

        ll_seg_mask_colored = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)
        ll_seg_mask_colored[ll_seg_mask > 0] = [0, 0, 255]  # Red for lane lines

        # ✅ แปลงเป็น BEV
        bev_combined = transform_to_bev(img_det, da_seg_mask_colored, ll_seg_mask_colored)

        # รวม Lane Line และ Drivable Area ใน Front View
        front_combined = cv2.addWeighted(img_det, 1, da_seg_mask_colored, 1, 0)
        front_combined = cv2.addWeighted(front_combined, 1, ll_seg_mask_colored, 1, 0)

        # Resize for visualization
        front_view_resized = cv2.resize(front_combined, (640, 720))
        bev_view_resized = cv2.resize(bev_combined, (640, 720))

        # รวมมุมมอง (Front View + BEV)
        combined_view = np.hstack((front_view_resized, bev_view_resized))

        # บันทึกวิดีโอ
        if vid_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter(os.path.join(opt.save_dir, "output.mp4"), fourcc, 20, (1280, 720))

        vid_writer.write(combined_view)

        # แสดงผล
        cv2.imshow("Combined View (Front + BEV)", combined_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if vid_writer:
        vid_writer.release()

    print(f"Results saved to {Path(opt.save_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOP/weights/End-to-end.pth', help='model.pth path')
    parser.add_argument('--source', type=str, default='/home/satoi/final.mp4', help='source video path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/satoi/python/result', help='directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(cfg, opt)
