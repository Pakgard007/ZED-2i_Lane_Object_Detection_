import pyzed.sl as sl
import cv2
import numpy as np

def capture_images():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER
    zed.open(init_params)

    image_left = sl.Mat()
    image_right = sl.Mat()

    print("กด SPACE เพื่อถ่ายภาพ หรือ ESC เพื่อออก")
    
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            imgL = image_left.get_data()
            imgR = image_right.get_data()

            cv2.imshow("Left Camera", imgL)
            cv2.imshow("Right Camera", imgR)

            key = cv2.waitKey(1)
            if key == 32:  # SPACE -> ถ่ายภาพ
                cv2.imwrite("edit-left20.jpg", imgL)
                cv2.imwrite("edit-right20.jpg", imgR)
                print("บันทึกภาพสำเร็จ!")

            elif key == 27:  # ESC -> ออก
                break

    zed.close()
    cv2.destroyAllWindows()

capture_images()
