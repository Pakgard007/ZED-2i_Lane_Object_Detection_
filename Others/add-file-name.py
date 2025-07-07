import os

# 📂 ตั้งค่าโฟลเดอร์ที่ต้องการเปลี่ยนชื่อไฟล์
folder_path = "/home/satoi/python/lane_line"
old_suffix = "_lane"  # 🔥 ส่วนที่ต้องการเปลี่ยน
new_suffix = "_drivable_area"  # 🆕 ส่วนใหม่ที่ต้องการแทนที่

# ตรวจสอบว่าโฟลเดอร์มีอยู่จริง
if not os.path.exists(folder_path):
    print(f"❌ โฟลเดอร์ '{folder_path}' ไม่พบ!")
    exit(1)

# ค้นหาไฟล์ทั้งหมดในโฟลเดอร์
files = os.listdir(folder_path)

# ตรวจสอบว่าในโฟลเดอร์มีไฟล์หรือไม่
if not files:
    print(f"⚠️ ไม่มีไฟล์ในโฟลเดอร์ '{folder_path}'")
    exit(1)

# เปลี่ยนชื่อไฟล์ที่มี "_lane" เท่านั้น
for filename in files:
    old_path = os.path.join(folder_path, filename)

    # ข้ามหากเป็นโฟลเดอร์
    if os.path.isdir(old_path):
        continue

    # ตรวจสอบว่าไฟล์มี "_lane" อยู่ในชื่อหรือไม่
    if old_suffix in filename:
        new_filename = filename.replace(old_suffix, new_suffix)  # แทนที่ _lane ด้วย _drivable_area
        new_path = os.path.join(folder_path, new_filename)

        # เปลี่ยนชื่อไฟล์
        os.rename(old_path, new_path)
        print(f"✅ เปลี่ยนชื่อไฟล์: {filename} ➝ {new_filename}")

print("\n🎉 เสร็จสิ้น! เปลี่ยนชื่อไฟล์ทั้งหมดแล้ว 🚀")
