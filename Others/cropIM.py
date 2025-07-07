import cv2

img = cv2.imread("right10.jpg")
roi = cv2.selectROI("Crop Image", img, fromCenter=False, showCrosshair=True)
cropped_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite("cropped_right10.jpg", cropped_img)
     