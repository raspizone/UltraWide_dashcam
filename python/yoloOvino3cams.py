import numpy as np
import cv2
from openvino.runtime import Core
import time

# Step 1 - Load the model using OpenVINO
core = Core()
model_path = "/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx"
compiled_model = core.compile_model(model=model_path, device_name="GPU")
infer_request = compiled_model.create_infer_request()
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# Step 2 - Initialize multiple cameras
cap1 = cv2.VideoCapture(4)  # Cámara 1
cap2 = cv2.VideoCapture(1)  # Cámara 2
cap3 = cv2.VideoCapture(2)  # Cámara 3

if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
    print("Error: Could not open one or more cameras.")
    exit()

# Load class names
class_list = []
with open("/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# FPS calculation setup
prev_time = time.time()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    
    if not ret1 or not ret2 or not ret3:
        print("Error: Unable to read frame from one or more cameras.")
        break

    # Step 3 - Prepare the frames
    frame1 = format_yolov5(frame1)  # Hacer la imagen cuadrada
    frame2 = format_yolov5(frame2)
    frame3 = format_yolov5(frame3)

    # Step 4 - Resize the frames to 640x640
    resized_frame1 = cv2.resize(frame1, (640, 640))
    resized_frame2 = cv2.resize(frame2, (640, 640))
    resized_frame3 = cv2.resize(frame3, (640, 640))

    # Step 5 - Perform inference on each individual frame
    frames = [resized_frame1, resized_frame2, resized_frame3]
    predictions = []

    for frame in frames:
        blob = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
        infer_request.infer(inputs={input_layer.any_name: blob})
        prediction = infer_request.get_output_tensor(output_layer.index).data
        predictions.append(prediction)

    # Step 6 - Process the predictions
    class_ids = []
    confidences = []
    boxes = []

    image_width, image_height, _ = resized_frame1.shape  # Todos los frames tienen el mismo tamaño
    x_factor = image_width / 640
    y_factor = image_height / 640

    for prediction in predictions:
        output_data = prediction[0]
        
        for r in range(25200):  # Número de posibles detecciones por imagen
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
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

    # If indexes is a tuple, extract the actual index array from it
    if len(indexes) > 0:
        indexes = indexes[0] if isinstance(indexes, tuple) else indexes

        # Iterate through the indexes
        for i in indexes:
            i = int(i)  # Ensure it's an integer index
            box = boxes[i]
            class_id = class_ids[i]
            cv2.rectangle(resized_frame1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 255), 2)
            cv2.rectangle(resized_frame1, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(resized_frame1, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Step 7 - Combine the frames horizontally after inference
    combined_frame = np.hstack((resized_frame1, resized_frame2, resized_frame3))

    # Step 8 - Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Combined Video with Detections", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()
