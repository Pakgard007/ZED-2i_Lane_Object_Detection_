import pyzed.sl as sl
import cv2
import numpy as np

# ----------------- 1. เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()

# ตั้งค่ากล้อง
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NONE  

# เปิดกล้อง
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างออบเจกต์สำหรับดึงภาพ
image = sl.Mat()

# ----------------- 2. ใช้ค่าคาลิเบรตจากไฟล์ -----------------
K = np.array([[537.679, 0, 635.247],  
              [0, 537.88, 360.651],  
              [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([-0.077093, 0.0675618, -0.028545, 0.00015007, -6.00634e-05], dtype=np.float32)

# ----------------- 3. ใช้ค่าการคาลิเบรต Rotation และ Translation -----------------
R = np.array([
    [0.9997,  0.0223,  0.0075],
    [-0.0223, 0.9997,  0.0022],
    [-0.0075, -0.0022,  0.9999]
], dtype=np.float32)

T = np.array([[0], [0], [-1.5]], dtype=np.float32)  # ระยะจากกล้องถึงพื้น (หน่วยเป็นเมตร)

n = np.array([[0, 0, 1]], dtype=np.float32).T  # พื้นอยู่ที่แกน Z
d = 1.5  # ระยะจากกล้องถึงพื้น

# คำนวณ Homography Matrix ที่ถูกต้อง
H = K @ (R - (T @ n.T) / d) @ np.linalg.inv(K)

print("Updated Homography Matrix (H):\n", H)

# ----------------- 4. ปรับค่าพิกัดใหม่ -----------------
src_pts = np.array([
    [400, 720],  # มุมซ้ายล่างของพื้น
    [880, 720],  # มุมขวาล่างของพื้น
    [500, 500],  # มุมซ้ายบนของพื้น
    [800, 500]   # มุมขวาบนของพื้น
], dtype=np.float32)  # ✅ ใช้ np.float32 

dst_pts = np.array([
    [300, 900],  # มุมซ้ายล่าง BEV
    [900, 900],  # มุมขวาล่าง BEV
    [300, 500],  # มุมซ้ายบน BEV
    [900, 500]   # มุมขวาบน BEV
], dtype=np.float32)  # ✅ ใช้ np.float32 

# ✅ แสดงค่าก่อน reshape
print("Before Reshaping")
print("src_pts dtype:", src_pts.dtype, "shape:", src_pts.shape, "values:", src_pts)
print("dst_pts dtype:", dst_pts.dtype, "shape:", dst_pts.shape, "values:", dst_pts)

# ✅ Reshape เป็น (4, 2) ตามที่ OpenCV ต้องการ
src_pts = src_pts.reshape(-1, 2).astype(np.float32, order='C')
dst_pts = dst_pts.reshape(-1, 2).astype(np.float32, order='C')

# ✅ แสดงค่าหลัง reshape
print("After Reshaping")
print("src_pts dtype:", src_pts.dtype, "shape:", src_pts.shape, "values:", src_pts)
print("dst_pts dtype:", dst_pts.dtype, "shape:", dst_pts.shape, "values:", dst_pts)

# ✅ ใช้ try-except กับ cv2.findHomography()
try:
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("✅ findHomography สำเร็จ")
except cv2.error as e:
    print("❌ cv2.findHomography ผิดพลาด:", e)
    exit()  # หยุดโปรแกรมถ้า Homography มีปัญหา

# ----------------- 5. อ่านภาพจากกล้องและแปลงเป็น BEV -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)

        # ✅ ตรวจสอบว่า image ไม่เป็นค่าว่าง
        if image.is_init():
            frame = image.get_data()[:, :, :3]  # ตัด Alpha channel ออก
            frame = np.array(frame, dtype=np.uint8)  # แปลงให้เป็น numpy array

            # ✅ แก้ปัญหา cv2.cvtColor() รับค่าไม่ได้
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

                # แก้ Distortion
                undistorted_frame = cv2.undistort(frame, K, dist_coeffs)

                # แปลงเป็น Bird's Eye View
                bev_image = cv2.warpPerspective(undistorted_frame, H, (1500, 800))

                # แสดงผลลัพธ์
                cv2.imshow("Front View - ZED 2i", frame)
                cv2.imshow("Bird's Eye View", bev_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 6. ปิดกล้องและปิดหน้าต่าง -----------------
zed.close()
cv2.destroyAllWindows()
