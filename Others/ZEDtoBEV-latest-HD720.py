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
              [0, 0, 1]])

dist_coeffs = np.array([-0.077093, 0.0675618, -0.028545, 0.00015007, -6.00634e-05])

# ----------------- 3. ใช้ค่าการคาลิเบรต Rotation -----------------
R = np.array([[ 9.99999957e-01, -2.92793227e-04,  6.70860728e-07],
            [ 2.92793996e-04,  9.99997332e-01, -2.29123790e-03],
            [ 0.00000000e+00,  2.29123800e-03,  9.99997375e-01]
])

T = np.array([[0], [0], [-7.0]])  # ระยะจากกล้องถึงพื้น (หน่วยเป็นเมตร)

n = np.array([[0, 0, 1]]).T  # พื้นอยู่ที่แกน Z
d = 7.0  # ระยะจากกล้องถึงพื้น

# คำนวณ Homography Matrix ที่ถูกต้อง
H = K @ (R - (T @ n.T) / d) @ np.linalg.inv(K)

print("Updated Homography Matrix (H):\n", H)

# ----------------- 4. ปรับค่าพิกัดใหม่ -----------------
src_pts = np.array([
    [400, 720],  # มุมซ้ายล่างของพื้น
    [880, 720],  # มุมขวาล่างของพื้น
    [500, 450],  # ปรับมุมซ้ายบนของพื้นให้สูงขึ้น
    [800, 450]   # ปรับมุมขวาบนของพื้นให้สูงขึ้น
], dtype=np.float32)

dst_pts = np.array([
    [300, 850],  # ขยับมุมซ้ายล่างของ BEV ขึ้น
    [900, 850],  # ขยับมุมขวาล่างของ BEV ขึ้น
    [300, 400],  # ปรับมุมซ้ายบนให้สูงขึ้น
    [900, 400]   # ปรับมุมขวาบนให้สูงขึ้น
], dtype=np.float32)


# คำนวณ Homography Matrix ใหม่
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# ----------------- 5. อ่านภาพจากกล้องและแปลงเป็น BEV -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, K, dist_coeffs)

        # แปลงเป็น Bird's Eye View
        # bev_image = cv2.warpPerspective(undistorted_frame, H, (1500, 800))
        bev_image = cv2.warpPerspective(undistorted_frame, H, (1800, 900))
        
        # แสดงผลลัพธ์
        cv2.imshow("Front View - ZED 2i", frame)
        cv2.imshow("Bird's Eye View", bev_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 6. ปิดกล้องและปิดหน้าต่าง -----------------
zed.close()
cv2.destroyAllWindows()
