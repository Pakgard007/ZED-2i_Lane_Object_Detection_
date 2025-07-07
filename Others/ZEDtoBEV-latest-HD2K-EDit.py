import pyzed.sl as sl
import cv2
import numpy as np

# ----------------- 1. เปิดกล้อง ZED 2i -----------------
zed = sl.Camera()

# ตั้งค่ากล้อง
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K  # ปรับเป็น 2K
init_params.camera_fps = 15  # 2K รองรับสูงสุด 15 FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE  

# เปิดกล้อง
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างออบเจกต์สำหรับดึงภาพ
image = sl.Mat()

# ----------------- 2. ใช้ค่าคาลิเบรตจากไฟล์ (2K) -----------------
K = np.array([[1075.36, 0, 1097.49],  
              [0, 1075.76, 625.303],  
              [0, 0, 1]])

dist_coeffs = np.array([-0.077093, 0.0675618, -0.0285450, 0.00015007, -6.00634e-05])

# ----------------- 3. ใช้ค่าการคาลิเบรต Rotation และ Translation -----------------
R = np.array([[ 9.99999957e-01, -2.92793227e-04,  6.70860728e-07],
            [ 2.92793996e-04,  9.99997332e-01, -2.29123790e-03],
            [ 0.00000000e+00,  2.29123800e-03,  9.99997375e-01]
])

T = np.array([[0], [0], [-7.0]])  # ระยะจากกล้องถึงพื้น (หน่วยเป็นเมตร) ปรับเป็น 7m เพื่อยกมุมมองสูงขึ้น

n = np.array([[0, 0, 1]]).T  # พื้นอยู่ที่แกน Z
d = 7  # ระยะจากกล้องถึงพื้น

# คำนวณ Homography Matrix ที่ถูกต้อง
H = K @ (R - (T @ n.T) / d) @ np.linalg.inv(K)

print("Updated Homography Matrix (H):\n", H)

# ----------------- 4. ปรับค่าพิกัดใหม่ -----------------
src_pts = np.array([
    [750, 900],  # มุมซ้ายล่างของพื้น (2K)
    [1600, 900],  # มุมขวาล่างของพื้น
    [1050, 400],  # มุมซ้ายบนของพื้น
    [1200, 400]   # มุมขวาบนของพื้น
], dtype=np.float32)

dst_pts = np.array([
    [600, 1400],  # มุมซ้ายล่างของ BEV
    [1400, 1400],  # มุมขวาล่างของ BEV
    [600, 800],  # มุมซ้ายบนของ BEV
    [1400, 800]   # มุมขวาบนของ BEV
], dtype=np.float32)

# คำนวณ Homography Matrix ใหม่
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)


# ----------------- 5. ฟังก์ชันวาดจุดพิกัด -----------------
def draw_points(image, points, color, labels=True):
    """
    วาดจุดพิกัดบนภาพและใส่หมายเลขจุด

    Args:
        image: ภาพที่ต้องการวาด
        points: จุดพิกัดที่ต้องการวาด
        color: สีของจุด (BGR)
        labels: True ถ้าต้องการใส่ตัวเลขกำกับจุด
    """
    for i, (x, y) in enumerate(points):
        cv2.circle(image, (int(x), int(y)), 10, color, -1)  # วาดวงกลมที่จุด
        if labels:
            cv2.putText(image, str(i+1), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ----------------- 6. อ่านภาพจากกล้องและแปลงเป็น BEV -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, K, dist_coeffs)

        # วาดจุด `src_pts` บนภาพต้นฉบับ (Front View)
        front_view = undistorted_frame.copy()
        draw_points(front_view, src_pts, (0, 0, 255))  # 🔴 วาดจุดสีแดงใน Front View

        # แปลงเป็น Bird's Eye View
        bev_image = cv2.warpPerspective(undistorted_frame, H, (1800, 900))

        # วาดจุด `dst_pts` บนภาพ BEV
        draw_points(bev_image, dst_pts, (255, 0, 0))  # 🔵 วาดจุดสีน้ำเงินใน BEV

        # แสดงผลลัพธ์
        cv2.imshow("Front View - ZED 2i (2K)", front_view)
        cv2.imshow("Bird's Eye View (2K)", bev_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 7. ปิดกล้องและปิดหน้าต่าง -----------------
zed.close()
cv2.destroyAllWindows()
