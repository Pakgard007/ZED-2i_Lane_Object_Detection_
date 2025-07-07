import numpy as np
import cv2

# ----------------- 1️⃣ กำหนดค่า Intrinsic Parameters -----------------
K = np.array([[537.679, 0, 635.247],  
              [0, 537.88, 360.651],  
              [0, 0, 1]])

# ----------------- 2️⃣ ใช้ค่า Rotation และ Translation ที่คุณให้มา -----------------
R_left = np.array([[ 0.99963516,  0.02344795, -0.00409754],
                   [-0.00606465,  0.41676231,  0.90874696],
                   [ 0.0230085,  -0.90845591,  0.41673925]])

T_left = np.array([[-0.44385514],
                   [ 0.53549433],
                   [ 3.19085962]])

# ----------------- 3️⃣ กำหนดจุด Checkerboard 6x4 (จุดตัดทั้งหมด) -----------------
rows = 6  # จุดตัดแนวนอน
cols = 4  # จุดตัดแนวตั้ง
square_size = 0.20  # ขนาดช่องของ Checkerboard (50 มม. หรือ 5 ซม.)

# สร้างพิกัด 3D ของทุกจุดตัด
checkerboard_points = np.zeros((rows * cols, 3), dtype=np.float32)
checkerboard_points[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

# ----------------- 4️⃣ โปรเจกต์จุด 3D ไปยังภาพ 2D -----------------
projected_points, _ = cv2.projectPoints(checkerboard_points, cv2.Rodrigues(R_left)[0], T_left, K, None)

# ----------------- 5️⃣ คืนค่าพิกัดจุดที่โปรเจกต์ได้ -----------------
projected_points.reshape(-1, 2)

img = cv2.imread("edit-left05.jpg")  # เปลี่ยนเป็นชื่อไฟล์ภาพของคุณ
for p in projected_points:
    x, y = int(p[0][0]), int(p[0][1])
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 🔵 วาดจุดสีเขียว
    
cv2.imshow("Projected Checkerboard - 6x4", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
