import torch
import cv2
import numpy as np

# Load the pre-trained YOLOP and YOLOv5 models
model_yolop = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True)
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # ใช้ 'yolov5s' เป็นเวอร์ชันเล็ก

# Function for lane detection
def lane_detection(frame, lane_output):
    if lane_output is None or len(lane_output.shape) != 4:
        print(f"Lane output shape is invalid: {lane_output.shape}")
        return frame

    # เลือกแชนเนลที่ต้องการ
    lane_output = lane_output[0, 1]  # ใช้ช่องที่ 1 สำหรับเลนซ้าย (หรือ 0 สำหรับเลนขวา)
    lane_output = lane_output.squeeze().cpu().numpy()

    if lane_output.size == 0 or len(lane_output.shape) != 2:
        print(f"Lane output shape is invalid after selection: {lane_output.shape}")
        return frame

    # Resize lane output ให้ตรงกับขนาดของเฟรม
    lane_output = cv2.resize(lane_output, (frame.shape[1], frame.shape[0]))

    # Apply thresholding for lane detection
    _, lane_binary = cv2.threshold(lane_output, 0.5, 255, cv2.THRESH_BINARY)
    lane_binary = np.uint8(lane_binary)

    # สร้าง mask เพื่อตัดส่วนที่ไม่ต้องการออก
    mask = np.zeros_like(lane_binary)
    height, width = lane_binary.shape
    poly = np.array([[ (0, height * 1 // 3), (width, height * 1 // 3), (width, height), (0, height) ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)

    # Apply mask to the lane binary
    masked_lane = cv2.bitwise_and(lane_binary, mask)

    # ใช้ Hough Line Transform สำหรับการตรวจจับเส้นเลน
    lines = cv2.HoughLinesP(masked_lane, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=20)

    # สร้างสำเนาของ frame เพื่อวาดเส้นลงไป
    lane_image = frame.copy()

    # วาดเส้นเลนลงในสำเนา lane_image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # สีแดงและความหนาของเส้น 2
    else:
        print("No lane detected.")

    # ทำ alpha blending เพื่อลดความเข้มของเส้นเลน
    alpha = 0.5  # ปรับความจางของเส้น (ค่าต่ำ = จางมากขึ้น)
    blended_frame = cv2.addWeighted(frame, 1 - alpha, lane_image, alpha, 0)

    return blended_frame

# Function for drivable area detection
def drivable_area_detection(frame, da_output):
    if da_output is None or not isinstance(da_output, torch.Tensor):
        print("No drivable area detected or da_output is not a valid tensor")
        return frame

    da_output = da_output.squeeze().cpu().numpy()

    if da_output.size == 0:
        print("Drivable area output is empty")
        return frame

    da_output = da_output[0] if da_output.ndim == 3 else da_output

    if da_output.shape[0] == 0 or da_output.shape[1] == 0:
        print("Drivable area output dimensions are invalid")
        return frame

    da_output = cv2.resize(da_output, (frame.shape[1], frame.shape[0]))

    # Apply threshold for drivable area
    _, da_binary = cv2.threshold(da_output, 0.6, 255, cv2.THRESH_BINARY)
    da_binary = np.uint8(da_binary)

    # สร้างหน้ากากสีเขียว (เช่น สีเขียวที่มีความเข้มน้อยลง)
    green_mask = np.zeros_like(frame)
    green_mask[da_binary == 0] = [0, 255, 0]  # ใส่สีเขียวในพื้นที่ขับขี่

    # ปรับระดับความจาง (alpha blending) โดยให้สีเขียวจางลง
    alpha = 0.3  # ความจางของสีเขียว (สามารถปรับค่าได้)
    blended_frame = cv2.addWeighted(frame, 1 - alpha, green_mask, alpha, 0)
    
    # ใช้การ Morphological Transformations เพื่อลด Noise
    kernel = np.ones((5, 5), np.uint8)
    da_binary = cv2.morphologyEx(da_binary, cv2.MORPH_OPEN, kernel)
    
    return blended_frame

# Function for object detection using YOLOv5
def object_detection(frame, model_yolov5):
    # ทำการตรวจจับวัตถุผ่านโมเดล YOLOv5
    results = model_yolov5(frame)

    # ดึงข้อมูล bounding box, ค่าความเชื่อมั่น และ class label
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # วาดกรอบสี่เหลี่ยมรอบวัตถุ
    n = len(labels)  # จำนวนวัตถุที่ตรวจพบ
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.6:  # confidence threshold 0.6
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 51, 153), 2)
            label = f'{model_yolov5.names[int(labels[i])]} {row[4]:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

# Start capturing video from the webcam (use 0 for the first webcam device)
cap = cv2.VideoCapture(0)

# Frame skipping and resolution settings
frame_skip = 8  # Skip every 6 frames
frame_count = 0
resize_width, resize_height = 480, 480  # Lower resolution to 320x320

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce load
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Split the frame into left and right parts (assuming two frames are side by side)
    height, width, _ = frame.shape
    half_width = width // 2

    # Use only the left frame
    left_frame = frame[:, :half_width]  # Select the left side of the frame
    
    # Resize the frame for the models
    resized_frame = cv2.resize(left_frame, (resize_width, resize_height))

    # Prepare the image for the YOLOP model
    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Run the YOLOP model
    with torch.no_grad():
        det_out, da_out, ll_out = model_yolop(img)

    # Lane detection
    lane_image = lane_detection(resized_frame, ll_out)

    # Drivable area detection
    da_image = drivable_area_detection(lane_image, da_out)

    # Object detection using YOLOv5
    final_image = object_detection(da_image, model_yolov5)

    # Display the results
    cv2.imshow('YOLOP and YOLOv5: Object, Lane, and Drivable Area Detection', final_image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
