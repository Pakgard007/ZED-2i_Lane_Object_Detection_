import torch
import cv2
import numpy as np

# Load YOLOv5 model from the official GitHub repository
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to calculate the distance of detected objects based on bounding box size
def calculate_distance(bbox):
    width = bbox[2] - bbox[0]
    distance = 1000 / width  # Approximation based on bounding box width
    return distance

# Function to adjust speed based on object type
def adjust_speed(label, distance):
    if label == "person":
        return "Person detected", "Slow down: Person ahead"
    elif label == "car":
        if distance > 10:
            return "Car detected", "Maintain speed: Car far away"
        elif 5 < distance <= 10:
            return "Car detected", "Slow down: Car ahead"
        else:
            return "Car detected", "Slow down significantly: Car very close"
    elif label in ["cone", "small obstacle"]:
        return "Obstacle detected", "Slow down: Obstacle ahead"
    else:
        return f"{label} detected", "Maintain normal speed"

# Function to add background for text
def draw_text_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, font_thickness=2, text_color=(0, 0, 255), bg_color=(0, 0, 0), alpha=0.5):
    # Get the size of the text box
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    x, y = pos

    # Create a filled rectangle for the background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)

    # Blend the overlay with the image to make the background semi-transparent
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add the text on top of the background
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)

# Open the webcam for video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Split the frame into left and right parts (assuming stereo camera)
    height, width, _ = frame.shape
    half_width = width // 2

    # Choose to display either left or right frame
    left_frame = frame[:, :half_width]   # Left frame
    right_frame = frame[:, half_width:]  # Right frame

    # Uncomment one of the lines below to select the frame to display
    selected_frame = left_frame  # To display only the left frame
    # selected_frame = right_frame  # To display only the right frame

    # Perform object detection using YOLOv5
    results = model_yolov5(frame)

    # Retrieve object detection results from YOLOv5
    detections = results.pred[0]  # Access the first frame of the predictions

    # Variables to store detection and speed information
    detection_info = "No objects detected"
    speed_info = "Speed: Normal"

    # Iterate through detected objects
    for det in detections:
        x1, y1, x2, y2, confidence, cls = det.tolist()
        label = model_yolov5.names[int(cls)]  # Get label of the class

        # Only display bounding box if confidence is greater than 0.6
        if confidence > 0.6:
            # Calculate the distance of the object
            distance = calculate_distance([x1, y1, x2, y2])

            # Adjust speed based on the object detected
            detection_message, speed_message = adjust_speed(label, distance)

            # Update speed and detection information to display later
            detection_info = f"{detection_message}, Confidence: {confidence:.2f}"
            speed_info = speed_message

            # Draw the bounding box and label on the frame, showing the distance
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 51, 153), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}, Dist: {distance:.2f} m', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw the two lines of text with background
    draw_text_with_background(selected_frame, detection_info, (10, 30))  # Detection info and confidence
    draw_text_with_background(selected_frame, speed_info, (10, 60))  # Speed decision

    # Display the frame with object detection
    cv2.imshow('YOLOv5 Object Detection', selected_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
