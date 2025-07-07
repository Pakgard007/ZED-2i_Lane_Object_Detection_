import os
import json

# พาธโฟลเดอร์ต้นทางและปลายทาง
input_folder = "/home/satoi/json-pp-finished"
output_folder = "/home/satoi/json-perfect"

# สร้างโฟลเดอร์ปลายทางหากยังไม่มี
os.makedirs(output_folder, exist_ok=True)

# ค้นหาไฟล์ JSON ทั้งหมดในโฟลเดอร์ต้นทาง
json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

if not json_files:
    print(f" ไม่พบไฟล์ JSON ในโฟลเดอร์: {input_folder}")
    exit(1)

for json_file in json_files:
    input_json_path = os.path.join(input_folder, json_file)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ ไม่สามารถอ่านไฟล์: {json_file} (ไฟล์ JSON อาจเสียหาย)")
            continue

    # ดึงชื่อภาพจาก JSON
    image_name = data.get("imagePath", "unknown.jpg")

    # ดึงขนาดภาพ
    image_width = data.get("imageWidth", 1280)
    image_height = data.get("imageHeight", 720)

    # จัดรูปแบบ objects (bounding box)
    objects = []
    lanes = []
    drivable_area = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower()
        points = shape.get("points", [])

        if label in ["person", "car", "bus", "bike", "truck"]:  # หมวดหมู่ Object ที่ใช้ bbox
            if len(points) == 2:
                bbox = [points[0][0], points[0][1], points[1][0], points[1][1]]
                objects.append({"category": label, "bbox": bbox, "mask": None})

        elif label == "lane":  # หมวดหมู่เส้นเลน
            lanes.append({"points": points})

        elif label == "drivable_area":  # หมวดหมู่พื้นที่ขับขี่
            drivable_area.append({"points": points})

    # สร้างโครงสร้าง JSON ใหม่
    new_data = {
        "image": f"../captured_images/{image_name}",
        "image_width": image_width,
        "image_height": image_height,
        "objects": objects,
        "lanes": lanes,
        "drivable_area": drivable_area,
        "frames": [{"objects": []}]
    }

    # บันทึกเป็นไฟล์ JSON ใหม่
    output_json_path = os.path.join(output_folder, json_file)
    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, indent=4)

    print(f"✅ บันทึกไฟล์: {output_json_path}")

print(f"\n🎉 แปลงไฟล์ JSON เสร็จสมบูรณ์! บันทึกที่ {output_folder}")
