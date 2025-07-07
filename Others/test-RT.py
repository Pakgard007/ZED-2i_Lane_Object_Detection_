import numpy as np

# ค่า R ที่ได้จาก solvePnP()
R = np.array([[0.99963516,  0.02344795, -0.00409754],
              [-0.00606465,  0.41676231,  0.90874696],
              [0.0230085,  -0.90845591,  0.41673925]])

# ตรวจสอบว่า R @ R.T ≈ I
identity_check = np.allclose(R @ R.T, np.eye(3), atol=1e-5)
det_R = np.linalg.det(R)

print("✅ Determinant of R (ควรใกล้ 1):", det_R)

T = np.array([[-0.44385514], [0.53549433], [3.19085962]])  # ค่าที่ได้จาก solvePnP

# หาความยาวของเวกเตอร์ T (ระยะห่างจากกล้องถึง Checkerboard)
T_magnitude = np.linalg.norm(T)

print("✅ ขนาดของ T (ควรอยู่ในช่วง 0.1 - 10 เมตร):", T_magnitude)

