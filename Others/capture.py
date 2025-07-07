import cv2
import os
import time
import pyzed.sl as sl  # สำหรับใช้งาน ZED SDK

# กำหนดโฟลเดอร์ที่จะเก็บภาพ
output_folder = "captured_images"
os.makedirs(output_folder, exist_ok=True)

# ตั้งค่ากล้อง ZED 2i
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Ultra HD (4K)
init_params.camera_fps = 30  # ตั้งค่า FPS
init_params.coordinate_units = sl.UNIT.METER  # ใช้หน่วยเมตร

# เปิดกล้อง ZED 2i
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ไม่สามารถเปิดกล้อง ZED 2i: {status}")
    exit(1)

# สร้าง Mat สำหรับเก็บภาพจาก ZED
image_zed = sl.Mat()

# ตัวแปรสำหรับการเก็บภาพ
frame_count = 0
start_time = time.time()

while True:
    # ดึงภาพจากกล้อง ZED
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)  # ดึงภาพจากกล้อง

        # ดึงข้อมูลภาพดิบจาก ZED SDK
        frame = image_zed.get_data()[:, :, :3]

        # หากสีเพี้ยน ลองแปลงสีแบบกลับไปมา
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ทดสอบแปลง

        # แสดงภาพสดจากกล้อง ZED
        cv2.imshow("ZED 2i Capture", frame)

        # กด 's' เพื่อถ่ายภาพและบันทึก
        if cv2.waitKey(1) & 0xFF == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"{output_folder}/image_{timestamp}.jpg"
            
            # บันทึกภาพ
            cv2.imwrite(file_name, frame)
            print(f"บันทึกภาพเป็น {file_name}")

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ปิดการใช้งานกล้อง ZED และปิดหน้าต่าง
zed.close()
cv2.destroyAllWindows()
