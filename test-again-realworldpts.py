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

# ----------------- 2. โหลดค่าคาลิเบรตจากไฟล์ -----------------
calib_data = np.load("camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = np.array([
    -0.07709319689152208, 0.06756180189133752, 0.00015006759935512075,
    -6.006342505065124e-05, -0.028545020615709165
])

extrinsic_data = np.load("extrinsic_parameters.npz")
R = extrinsic_data["R_left"]
T = extrinsic_data["T_left"]

print("✅ Loaded Intrinsic and Extrinsic Parameters!")

# ----------------- 3. คำนวณ Homography Matrix -----------------
n = np.array([[0, 0, 1]]).T  # พื้นอยู่ที่แกน Z
d = 1.8  # ระยะจากกล้องถึงพื้น (หน่วยเป็นเมตร)

# คำนวณ Homography Matrix
H = mtxL @ (R - (T @ n.T) / d) @ np.linalg.inv(mtxL)
print("Updated Homography Matrix (H):\n", H)

# ----------------- 4. ฟังก์ชันแปลงพิกเซลเป็น World Coordinate -----------------
def pixel_to_world(u, v, depth, K, R, T):
    """ แปลงพิกัดพิกเซล (u, v) เป็น World Coordinate """
    uv_1 = np.array([u, v, 1.0]).reshape(3, 1)
    Xc = np.linalg.inv(K) @ (uv_1 * depth)  # คำนวณจุดใน Camera Coordinate
    Xw = np.linalg.inv(R) @ (Xc - T)  # คำนวณจุดใน World Coordinate
    return Xw.flatten()

# ----------------- 5. กำหนดพิกัดจุดที่ใช้ปรับ BEV -----------------
src_pts = np.array([
    [300, 700],  # มุมซ้ายล่างของพื้น
    [1000, 700],  # มุมขวาล่างของพื้น
    [550, 300],  # ปรับมุมซ้ายบนของพื้นให้สูงขึ้น
    [750, 300]   # ปรับมุมขวาบนของพื้นให้สูงขึ้น
], dtype=np.float32)

dst_pts = np.array([
    [300, 850],  # ขยับมุมซ้ายล่างของ BEV ขึ้น
    [900, 850],  # ขยับมุมขวาล่างของ BEV ขึ้น
    [300, 400],  # ปรับมุมซ้ายบนให้สูงขึ้น
    [900, 400]   # ปรับมุมขวาบนให้สูงขึ้น
], dtype=np.float32)

# คำนวณ Homography Matrix ใหม่จากค่าพิกัดที่ปรับปรุง
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# ----------------- 6. ฟังก์ชันวาดจุดพิกัด -----------------
def draw_points(image, points, color, labels=True):
    """ วาดจุดพิกัดบนภาพและใส่หมายเลขจุด """
    for i, (x, y) in enumerate(points):
        cv2.circle(image, (int(x), int(y)), 8, color, -1)
        if labels:
            cv2.putText(image, str(i+1), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ----------------- 7. อ่านภาพจากกล้องและแปลงเป็น BEV -----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # แปลงเป็น Bird's Eye View
        bev_image = cv2.warpPerspective(undistorted_frame, H, (1280, 720))

        # ปรับขนาดภาพให้เท่ากันเพื่อแสดงผล
        front_resized = cv2.resize(undistorted_frame, (640, 720))
        bev_resized = cv2.resize(bev_image, (640, 720))

        # รวมภาพแสดงผลพร้อมกัน
        combined_view = np.hstack((front_resized, bev_resized))

        # แสดงผลลัพธ์
        cv2.imshow("Front View (Left) | Bird's Eye View (Right)", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------- 8. ปิดกล้องและปิดหน้าต่าง -----------------
zed.close()
cv2.destroyAllWindows()
