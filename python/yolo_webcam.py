import numpy as np
import cv2
import time

# Step 1 - Load the model
net = cv2.dnn.readNet("yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx")

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# Step 2 - Initialize camera
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara por defecto; cambia si tienes múltiples cámaras.

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load class names
class_list = []
with open("yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# FPS calculation setup
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    # Step 3 - Prepare the frame
    input_image = format_yolov5(frame)  # Make the image square
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Step 4 - Process the predictions
    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])

    # Apply NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    for i in indexes.flatten():
        box = boxes[i]
        class_id = class_ids[i]
        cv2.rectangle(frame, box, (0, 255, 255), 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(frame, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Step 5 - Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Step 6 - Display the frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
