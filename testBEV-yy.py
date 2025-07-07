import cv2
import numpy as np
import os

def get_intrinsic_matrix():
    """
    สร้างเมทริกซ์ Intrinsic ของกล้อง (ใช้ค่าที่คุณให้มา)
    """
    K = np.array([
        [537.679,  0, 635.247],
        [0, 537.88, 360.651],
        [0,   0,  1]
    ])
    return K

def undistort_image(image_path, K, dist_coeffs):
    """
    แก้ไขความผิดเพี้ยนของเลนส์ในภาพ
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, dist_coeffs, None, new_K)
    return undistorted_img

def find_extrinsic_parameters(K, dist_coeffs, checkerboard_size, square_size, image_path):
    """
    คำนวณค่า Rotation (R) และ Translation (T) ของกล้องเมื่อเทียบกับ Checkerboard
    """
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # แปลงหน่วยจาก grid เป็นเมตร

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ตรวจจับ Checkerboard
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners, K, dist_coeffs)
        R, _ = cv2.Rodrigues(rvecs)

        print(f"✅ Extrinsic parameters for {image_path}:")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", tvecs)
        return R, tvecs
    else:
        print(f"❌ Checkerboard corners not detected in {image_path}.")
        return None, None

def warp_to_bev(image_path, K, R, T, checkerboard_size, square_size, output_size=(800, 800)):
    """
    แปลงภาพจากมุมกล้องให้เป็น Bird's Eye View (BEV)
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # พิกัดของ Checkerboard ในโลกจริง (World Coordinate)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # ตรวจจับ Checkerboard
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if not ret:
        print("❌ Checkerboard not detected, skipping BEV transformation.")
        return image

    # เลือก 4 จุดมุมของ Checkerboard
    src_pts = np.array([
        corners[0][0],  
        corners[checkerboard_size[0] - 1][0],  
        corners[-1][0],  
        corners[-checkerboard_size[0]][0]  
    ], dtype=np.float32)

    # กำหนดพิกัดปลายทางใน Bird's Eye View
    dst_pts = np.array([
        [0, 0],  
        [output_size[0] - 1, 0],  
        [output_size[0] - 1, output_size[1] - 1],  
        [0, output_size[1] - 1]  
    ], dtype=np.float32)

    # คำนวณ Homography Matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # ใช้ Homography Matrix เพื่อแปลงภาพ
    bev_image = cv2.warpPerspective(image, H, output_size)

    return bev_image

def main():
    # 🔹 **กำหนดค่าพารามิเตอร์ของกล้อง**
    K = get_intrinsic_matrix()
    
    # 🔹 **ค่าความผิดเพี้ยนของเลนส์**
    dist_coeffs = np.array([
    -0.07709319689152208,  # k1
     0.06756180189133752,  # k2
     0.00015006759935512075,  # p1 (tangential distortion)
    -6.006342505065124e-05,  # p2 (tangential distortion)
    -0.028545020615709165  # k3
])

    # 🔹 **กำหนดค่าของ Checkerboard**
    CHECKERBOARD = (6, 4)  # จุดตัด
    square_size = 0.20  # หน่วยเป็นเมตร

    # 🔹 **เลือกภาพที่ใช้คำนวณ Extrinsic Parameters**
    target_image = "edit-left01.jpg"

    # 🔹 **คำนวณค่าพารามิเตอร์ภายนอก**
    R, T = find_extrinsic_parameters(K, dist_coeffs, CHECKERBOARD, square_size, target_image)

    if R is not None and T is not None:
        # 🔹 **แปลงภาพเป็น Bird's Eye View**
        bev_image = warp_to_bev(target_image, K, R, T, CHECKERBOARD, square_size)

        # 🔹 **บันทึกภาพแทนการแสดงผล**
        output_path = "bev_output.jpg"
        cv2.imwrite(output_path, bev_image)
        print(f"✅ Bird's Eye View image saved as {output_path}")

if __name__ == "__main__":
    main()
