import numpy as np

# ----------------- 1️⃣ ใช้ค่า Rotation และ Translation ที่ได้จาก SolvePnP -----------------
R_left = np.array([[ 0.99963516,  0.02344795, -0.00409754],
                   [-0.00606465,  0.41676231,  0.90874696],
                   [ 0.0230085,  -0.90845591,  0.41673925]])

T_left = np.array([[-0.44385514],
                   [ 0.53549433],
                   [ 3.19085962]])

# ----------------- 2️⃣ คำนวณตำแหน่งของกล้องในระบบพิกัดโลก -----------------
# ใช้สูตร Camera Position = - R^T * T
camera_position = -np.dot(R_left.T, T_left)

# แสดงผลลัพธ์
print("ตำแหน่งของกล้องใน World Coordinate System:\n", camera_position)
