from PIL import Image
import os

# โฟลเดอร์ที่เก็บรูปภาพ
input_folder = '/home/satoi/python/captured_images'
output_folder = '/home/satoi/prepare_train'

# ตรวจสอบว่ามีโฟลเดอร์สำหรับผลลัพธ์หรือไม่ ถ้าไม่มีให้สร้างใหม่
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ขนาดเป้าหมาย
target_size = (1280, 720)

# อ่านไฟล์ทั้งหมดในโฟลเดอร์
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # เปิดรูปภาพ
        with Image.open(input_path) as img:
            # แปลงขนาดรูปภาพ
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            # บันทึกรูปภาพที่แปลงขนาดแล้ว
            resized_img.save(output_path)
            print(f"Resized and saved: {output_path}")

print("All images have been resized successfully!")
