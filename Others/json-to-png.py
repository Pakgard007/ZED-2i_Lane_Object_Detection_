import json
import cv2
import numpy as np
import os

def json_to_png_for_lane_and_drivable_area(json_file, image_shape, lane_output_folder, da_output_folder):
    # อ่านไฟล์ JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # ขนาดของภาพ (height, width)
    height, width = image_shape

    # สร้างแผนที่มาสก์สำหรับ Lane และ Drivable Area
    lane_mask = np.zeros((height, width), dtype=np.uint8)
    da_mask = np.zeros((height, width), dtype=np.uint8)

    # วนลูปผ่านทุก shape (polygon หรือ bounding box) ในไฟล์ JSON
    for shape in data['shapes']:
        label = shape['label']  # เลเบลของการทำ annotation (เช่น 'lane', 'drivable_area')
        points = shape['points']  # จุดของ polygon

        # แปลงจุดเป็น numpy array
        polygon = np.array(points, np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        if label == "lane":
            # เติมสีขาว (255) ในพื้นที่ของเส้นเลน
            cv2.fillPoly(lane_mask, [polygon], 255)
        elif label == "drivable_area":
            # เติมสีขาว (255) ในพื้นที่ของพื้นที่ขับขี่ได้
            cv2.fillPoly(da_mask, [polygon], 255)

    # ตรวจสอบว่าโฟลเดอร์ 'lane_line' และ 'drivable_area' มีอยู่หรือไม่
    os.makedirs(lane_output_folder, exist_ok=True)
    os.makedirs(da_output_folder, exist_ok=True)

    # ตั้งชื่อไฟล์ที่บันทึก
    lane_output_path = os.path.join(lane_output_folder, os.path.basename(json_file).replace('.json', '_lane.png'))
    da_output_path = os.path.join(da_output_folder, os.path.basename(json_file).replace('.json', '_drivable_area.png'))

    # บันทึกผลลัพธ์เป็นไฟล์ PNG สำหรับเส้นเลน
    cv2.imwrite(lane_output_path, lane_mask)
    print(f"บันทึกไฟล์ Lane PNG ที่ {lane_output_path}")

    # บันทึกผลลัพธ์เป็นไฟล์ PNG สำหรับพื้นที่ขับขี่ได้
    cv2.imwrite(da_output_path, da_mask)
    print(f"บันทึกไฟล์ Drivable Area PNG ที่ {da_output_path}")

# ตัวอย่างการใช้งาน
json_file = "/home/satoi/json-pp/image_20250124_143458.json"  # ปรับเป็นพาธไฟล์ JSON ของคุณ
image_shape = (720, 1280)  # ขนาดของภาพ (height, width)
lane_output_folder = "lane_line"  # โฟลเดอร์ที่ใช้เก็บไฟล์ Lane PNG
da_output_folder = "drivable_area"  # โฟลเดอร์ที่ใช้เก็บไฟล์ Drivable Area PNG

# แปลงไฟล์ JSON เป็น 2 PNG (Lane และ Drivable Area Mask)
json_to_png_for_lane_and_drivable_area(json_file, image_shape, lane_output_folder, da_output_folder)
