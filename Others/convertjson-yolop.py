import json
import os

def add_frames_key(input_json, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    # ตรวจสอบว่าไฟล์ไม่มีคีย์ 'frames'
    if 'frames' not in data:
        # สร้างโครงสร้างใหม่ที่มีคีย์ 'frames'
        data['frames'] = [{
            "objects": []
        }]

        # เพิ่มข้อมูลลงในคีย์ 'frames'
        for shape in data.get('shapes', []):
            label = shape['label']
            points = shape['points']

            # แปลงข้อมูล annotation
            if label == 'lane':  # สำหรับ Lane Detection
                data['frames'][0]['objects'].append({
                    "category": label,
                    "points": points
                })
            elif label == 'drivable_area':  # สำหรับ Drivable Area
                data['frames'][0]['objects'].append({
                    "category": label,
                    "points": points
                })
            else:  # สำหรับ Object Detection (Bounding Box)
                x_min = min([p[0] for p in points])
                y_min = min([p[1] for p in points])
                x_max = max([p[0] for p in points])
                y_max = max([p[1] for p in points])

                data['frames'][0]['objects'].append({
                    "category": label,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "mask": None
                })

        # บันทึกไฟล์ที่มีคีย์ 'frames' เพิ่มเข้าไป
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)

def process_all_json_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):  # ตรวจสอบเฉพาะไฟล์ .json
            input_json = os.path.join(input_folder, filename)
            output_json = os.path.join(output_folder, filename)

            # เพิ่มคีย์ 'frames' ในไฟล์ JSON
            add_frames_key(input_json, output_json)
            print(f"เพิ่มคีย์ 'frames' ในไฟล์ {filename} เสร็จเรียบร้อย!")

# ตัวอย่างการใช้งาน
input_folder = '/home/satoi/json-pp'  # โฟลเดอร์ที่เก็บไฟล์ .json ที่ต้องการแปลง
output_folder = '/home/satoi/json-pp-finished'    # โฟลเดอร์ที่ต้องการเก็บไฟล์ .json ที่มีคีย์ 'frames'
process_all_json_in_folder(input_folder, output_folder)
