import cv2
import numpy as np

# 📌 โหลดวิดีโอ
vidcap = cv2.VideoCapture("/home/satoi/final.mp4")

# ตรวจสอบว่าเปิดวิดีโอสำเร็จหรือไม่
if not vidcap.isOpened():
    print("Error: ไม่สามารถเปิดวิดีโอได้!")
    exit()

while True:
    success, image = vidcap.read()
    if not success:
        print("End of video.")
        break  # ออกจากลูปเมื่อถึงจบวิดีโอ

    # ตรวจสอบว่า image มีค่าหรือไม่ก่อน resize
    if image is None:
        print("Warning: อ่านเฟรมไม่ได้ ข้ามไปเฟรมถัดไป")
        continue

    frame = cv2.resize(image, (640, 480))

    # ✅ Perspective Transformation
    tl = (250, 300)  # Top-left (ซ้ายบน)
    bl = (150, 450)  # Bottom-left (ซ้ายล่าง)
    tr = (350, 300)  # Top-right (ขวาบน)  --> แก้ให้มีค่า y = 300 เท่ากับ TL
    br = (450, 450)  # Bottom-right (ขวาล่าง)  --> แก้ให้มีค่า y = 450 เท่ากับ BL
    

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # ✅ Perspective Transformation - Geometrical transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    cv2.imshow("Frame", frame)
    cv2.imshow("Transformed_Frame", transformed_frame)

    # กด ESC เพื่อออก
    if cv2.waitKey(1) == 27:
        break

# ปิดไฟล์วิดีโอ
vidcap.release()
cv2.destroyAllWindows()
