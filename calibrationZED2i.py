import cv2
import numpy as np
import glob

# โหลดค่าคาลิเบรตที่มีอยู่แล้ว
calib_data = np.load("/home/mag/satoi/python/camera_calibration.npz")
mtxL = calib_data["camera_matrix_L"]
distL = calib_data["dist_coeffs_L"]
mtxR = calib_data["camera_matrix_R"]
distR = calib_data["dist_coeffs_R"]

CHECKERBOARD_SIZE = (6, 4)
SQUARE_SIZE = 0.20  # ขนาดช่อง (เมตร)

# สร้างพิกัด 3D ของ Checkerboard
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # ใช้ขนาดช่องจริง

# โหลดไฟล์ภาพจากกล้องซ้ายและขวา
images_left = sorted(glob.glob("edit-left*.jpg"))
images_right = sorted(glob.glob("edit-right*.jpg"))

if len(images_left) == 0 or len(images_right) == 0:
    print("❌ ไม่พบภาพจากกล้องซ้ายหรือขวา!")
    exit()

if len(images_left) != len(images_right):
    print("❌ จำนวนภาพจากกล้องซ้ายและขวาไม่เท่ากัน!")
    exit()

R_left_list, T_left_list = [], []
R_right_list, T_right_list = [], []

for imgL_path, imgR_path in zip(images_left, images_right):
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # ค้นหา Checkerboard ในภาพซ้ายและขวา
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD_SIZE, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD_SIZE, None)

    if retL and retR:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if not retL or not retR:
        print(f"❌ ไม่พบ Checkerboard ในภาพ {imgL_path} หรือ {imgR_path}")
        continue

    # คำนวณค่า Extrinsic Parameters ของกล้องซ้าย
    ret, rvecL, tvecL = cv2.solvePnP(objp, cornersL, mtxL, distL)
    R_left, _ = cv2.Rodrigues(rvecL)
    T_left = tvecL
    # คำนวณค่า Extrinsic Parameters ของกล้องขวา
    ret, rvecR, tvecR = cv2.solvePnP(objp, cornersR, mtxR, distR)
    R_right, _ = cv2.Rodrigues(rvecR)
    T_right = tvecR
    R_left_list.append(R_left)
    T_left_list.append(T_left)
    R_right_list.append(R_right)
    T_right_list.append(T_right)

# คำนวณค่าเฉลี่ยของ Rotation และ Translation
R_left_avg = np.mean(R_left_list, axis=0)
T_left_avg = np.mean(T_left_list, axis=0)
R_right_avg = np.mean(R_right_list, axis=0)
T_right_avg = np.mean(T_right_list, axis=0)

print("✅ Rotation Matrix (Left Camera):\n", R_left_avg)
print("✅ Translation Vector (Left Camera):\n", T_left_avg)
print("✅ Rotation Matrix (Right Camera):\n", R_right_avg)
print("✅ Translation Vector (Right Camera):\n", T_right_avg)

# บันทึกค่าลงไฟล์
np.savez("extrinsic_parameters.npz", 
         R_left=R_left_avg, T_left=T_left_avg, 
         R_right=R_right_avg, T_right=T_right_avg)

print("✅ บันทึกค่า Extrinsic Parameters สำเร็จ!")
