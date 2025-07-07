import argparse
import os, sys
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision.transforms as transforms
import cv2
import pyzed.sl as sl

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.core.function import AverageMeter
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def detect_with_zed(cfg, opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    half = device.type != 'cpu'

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30  # Set FPS for Ultra mode
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)

    runtime_parameters = sl.RuntimeParameters()

    image = sl.Mat()
    depth = sl.Mat()
    print("Starting real-time detection...")

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img_np = image.get_data()[:, :, :3]  # Extract RGB image

            img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (opt.img_size, opt.img_size))
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_tensor = img_tensor.half() if half else img_tensor.float()

            # Inference
            t1 = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img_tensor)
            t2 = time_synchronized()

            inf_out, _ = det_out
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
            det = det_pred[0]

            da_seg_mask = torch.nn.functional.interpolate(da_seg_out, scale_factor=1, mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

            ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=1, mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

            # Resize segmentation masks to match original image dimensions
            da_seg_mask_resized = cv2.resize(da_seg_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask_resized = cv2.resize(ll_seg_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            img_result = show_seg_result(img_np, (da_seg_mask_resized, ll_seg_mask_resized), _, _, is_demo=True)

            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img_result.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{conf:.2f}"
                    plot_one_box(xyxy, img_result, label=label, color=(255, 0, 0), line_thickness=2)

            cv2.imshow("ZED YOLOP Real-Time", img_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOP/weights/End-to-end.pth', help='model.pth path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='device to use: cpu or cuda')
    opt = parser.parse_args()

    with torch.no_grad():
        detect_with_zed(cfg, opt)
