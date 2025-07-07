import numpy as np

# ค่าการหมุนจากไฟล์คาลิเบรต (หน่วยเป็นเรเดียน)
RX_HD = 0.00229124  # Rotation around X-axis
RY_HD = 0.0         # ไม่มีค่าจากคาลิเบรต จึงตั้งเป็น 0
RZ_HD = 0.000292794 # Rotation around Z-axis

# สร้าง Rotation Matrix สำหรับแต่ละแกน
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(RX_HD), -np.sin(RX_HD)],
    [0, np.sin(RX_HD), np.cos(RX_HD)]
])

Ry = np.array([
    [np.cos(RY_HD), 0, np.sin(RY_HD)],
    [0, 1, 0],
    [-np.sin(RY_HD), 0, np.cos(RY_HD)]
])

Rz = np.array([
    [np.cos(RZ_HD), -np.sin(RZ_HD), 0],
    [np.sin(RZ_HD), np.cos(RZ_HD), 0],
    [0, 0, 1]
])

# คำนวณ R ใหม่โดยใช้ Rx, Ry และ Rz
R_new = np.dot(Rz, np.dot(Ry, Rx))

# แสดงค่า R ที่คำนวณได้
print("Rotation Matrix (R) ที่ได้จาก RX และ RZ เท่านั้น:\n", R_new)
