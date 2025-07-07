import cv2
import numpy as np
import pyzed.sl as sl

# ---------------- 1️⃣ เปิดกล้อง ZED 2i ----------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ไม่สามารถเปิดกล้องได้!")
    exit()

image_left = sl.Mat()
depth_map = sl.Mat()

# ✅ โหลดค่าคาลิเบรตของกล้อง
calib_data = np.load("camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = np.array([-0.077093, 0.067562, 0.000150, -0.000060, -0.028545])

extrinsic_data = np.load("extrinsic_parameters.npz")
R = extrinsic_data["R_left"]
T = extrinsic_data["T_left"]

# ✅ ปรับค่า d เพื่อให้ BEV ถูกต้อง
d = 1.5  # ปรับระยะจากกล้องถึงพื้น

# ✅ คำนวณ Homography Matrix
n = np.array([[0, 0, 1]]).T
H = mtxL @ (R - (T @ n.T) / d) @ np.linalg.inv(mtxL)

# ✅ แก้ Homography Matrix เพื่อป้องกันภาพกลับด้าน
H[:, 1] *= -1  # กลับแกนที่สามแทน
# แก้ทิศทางของแกน Y
print("✅ Updated Homography Matrix (H):\n", H)

# ---------------- 2️⃣ ฟังก์ชันแปลงเป็น BEV ----------------
def transform_to_bev(image, H):
    """ แปลงภาพให้เป็น Bird's Eye View """
    h, w = image.shape[:2]
    bev_image = cv2.warpPerspective(image, H, (1200, 1200))
    bev_image = cv2.flip(bev_image, 0)  # Flip แนวตั้ง
    
    # ✅ Debug ค่า BEV
    print("Min BEV Pixel:", np.min(bev_image))
    print("Max BEV Pixel:", np.max(bev_image))

    # ✅ กลับด้าน BEV ถ้าจำเป็น
    bev_image = cv2.flip(bev_image, 0)  # Flip แนวตั้ง
    
    return bev_image

# ---------------- 3️⃣ วนลูปเพื่อแสดงผลแบบเรียลไทม์ ----------------
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        frame = image_left.get_data()[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        depth_data = depth_map.get_data()

        # ✅ Debug ค่า Depth
        print("Min Depth:", np.nanmin(depth_data))
        print("Max Depth:", np.nanmax(depth_data))
        print("Depth shape:", depth_data.shape)

        # ✅ Threshold ค่า Depth
        depth_data[np.isnan(depth_data)] = 0  # แปลง NaN เป็น 0
        depth_data[depth_data > 5] = 5  # จำกัดค่าสูงสุดที่ 5 เมตร

        # ✅ แสดง Depth Map ก่อน BEV
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=255.0/5.0), cv2.COLORMAP_JET)
        cv2.imshow("Depth Map", depth_colormap)

        # ✅ แก้ Distortion
        undistorted_frame = cv2.undistort(frame, mtxL, distL)

        # ✅ แปลงเป็น BEV
        bev_display = transform_to_bev(undistorted_frame, H)

        # ✅ ปรับขนาดให้เท่ากัน
        front_resized = cv2.resize(undistorted_frame, (640, 720))
        bev_resized = cv2.resize(bev_display, (640, 720))

        # ✅ รวมภาพ Side-by-Side
        combined_view = np.hstack((front_resized, bev_resized))

        # ✅ แสดงผล
        cv2.imshow("Front View | Bird's Eye View", combined_view)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---------------- 5️⃣ ปิดกล้องและหน้าต่าง ----------------
zed.close()
cv2.destroyAllWindows()
