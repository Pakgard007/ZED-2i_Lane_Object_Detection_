import numpy as np

# โหลดไฟล์
# calib_data = np.load("camera_calibration.npz")
calib_data = np.load("extrinsic_parameters.npz")

# แสดงชื่อของตัวแปรทั้งหมดในไฟล์
# print(calib_data.files)
# print("Camera Matrix (Left):\n", calib_data["camera_matrix_L"])
# print("Distortion Coefficients (Left):\n", calib_data["dist_coeffs_L"])

print(calib_data.files)
print("R-Left:\n", calib_data["R_left"])
print("T-Left:\n", calib_data["T_left"])
