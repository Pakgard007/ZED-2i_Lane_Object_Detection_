import pyzed.sl as sl
import cv2
import numpy as np

# ----------------- 1. เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()

# ตั้งค่ากล้อง
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # ความละเอียด 720p
init_params.camera_fps = 30  # 30 FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE  # ปิดการใช้ Depth Mode

# เปิดกล้อง
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างออบเจกต์สำหรับดึงภาพ
image = sl.Mat()

# ----------------- 2. ตั้งค่าการบันทึกวิดีโอ -----------------
frame_width = 1280
frame_height = 720
fps = 30  # FPS ที่ใช้บันทึก

# ตั้งชื่อไฟล์วิดีโอ (บันทึกที่โฟลเดอร์ปัจจุบัน)
video_filename = "record_video.mp4"

# กำหนด Codec และสร้าง VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ใช้ Codec MP4
video_out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

print(f"🎥 กำลังบันทึกวิดีโอที่ {frame_width}x{frame_height} @ {fps} FPS...")

# ----------------- 3. อ่านภาพจากกล้องและบันทึกวิดีโอ -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # ดึงภาพจากกล้อง
        frame = image.get_data()[:, :, :3]  # แปลงภาพเป็น NumPy array
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # แปลงเป็น BGR (OpenCV format)

        # แสดงผล
        cv2.imshow("ZED 2i - Live View", frame)

        # บันทึกวิดีโอ
        video_out.write(frame)

        # กด 'q' เพื่อหยุดบันทึก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ หยุดบันทึกวิดีโอ")
            break

# ----------------- 4. ปิดกล้องและบันทึกไฟล์ -----------------
zed.close()
video_out.release()
cv2.destroyAllWindows()

print(f"✅ วิดีโอบันทึกเสร็จแล้ว: {video_filename}")
