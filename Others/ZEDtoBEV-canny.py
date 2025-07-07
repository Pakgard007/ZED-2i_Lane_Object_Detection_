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

# ----------------- 2. ค่าการคาลิเบรต -----------------
K = np.array([[546.403, 0, 637.302],  
              [0, 546.565, 362.371],  
              [0, 0, 1]])

dist_coeffs = np.array([-0.057582, 0.033764, -0.000267, -0.000388, -0.014723])

# ----------------- 3. Homography Matrix -----------------
R = np.array([
    [0.9997,  0.0223,  0.0075],
    [-0.0223, 0.9997,  0.0022],
    [-0.0075, -0.0022,  0.9999]
])
T = np.array([[0], [0], [-1.5]])  
n = np.array([[0, 0, 1]]).T  
d = 1.5  

# คำนวณ Homography ที่ถูกต้อง
H = K @ (R - (T @ n.T) / d) @ np.linalg.inv(K)

print("Updated Homography Matrix (H):\n", H)

# ----------------- 4. ปรับค่าพิกัดใหม่ -----------------
src_pts = np.float32([
    [400, 720],  # มุมซ้ายล่างของพื้น
    [880, 720],  # มุมขวาล่างของพื้น
    [500, 500],  # มุมซ้ายบนของพื้น
    [800, 500]   # มุมขวาบนของพื้น
])

dst_pts = np.float32([
    [300, 900],  # มุมซ้ายล่าง BEV
    [900, 900],  # มุมขวาล่าง BEV
    [300, 500],  # มุมซ้ายบน BEV
    [900, 500]   # มุมขวาบน BEV
])

# ใช้ findHomography แทน
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
        bev_image = cv2.warpPerspective(undistorted_frame, H, (1500, 800))

        # ----------------- 6. ทำ Segmentation -----------------
        # 6.1 แปลงเป็น Grayscale
        gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)

        # 6.2 ใช้ Thresholding แยกพื้นกับวัตถุ
        _, segmented = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        # 6.3 ใช้ Canny Edge Detection
        edges = cv2.Canny(segmented, 50, 150)

        # ----------------- 7. แสดงผล -----------------
        cv2.imshow("Front View - ZED 2i", frame)
        cv2.imshow("Bird's Eye View", bev_image)
        cv2.imshow("Segmented BEV", segmented)
        cv2.imshow("Edges", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 8. ปิดกล้อง -----------------
zed.close()
cv2.destroyAllWindows()
