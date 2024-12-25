import numpy as np
import cv2
import time

# Cargar el modelo YOLOv5
net = cv2.dnn.readNet('/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx')

# Configuración de las rutas de los videos
video_paths = ["/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/python/video1.mp4", 
               "/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/python/video2.mp4", 
               "/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/python/video3.mp4"]  # Rutas de tus videos
captures = [cv2.VideoCapture(path) for path in video_paths]

# Dimensiones de la resolución final
final_width = 1280
final_height = 400
individual_width = final_width // len(video_paths)

# Lista de clases para la detección
class_list = []
with open("/home/andres/Descargas/yolov5/yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

# Función para redimensionar y recortar el frame
def crop_and_resize(frame, target_width, target_height):
    h, w, _ = frame.shape
    crop_left = (w - target_width) // 2
    crop_right = crop_left + target_width
    crop_top = (h - target_height) // 2
    crop_bottom = crop_top + target_height
    cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
    return cropped_frame

# Contador de fotogramas y detecciones previas
frame_counter = 0
analyze_interval = 1 # Analizar cada 30 frames
previous_detections = []  # Guardar detecciones previas

# Procesamiento de cada frame de video
while True:
    start_time = time.time()

    frames = []
    for capture in captures:
        ret, frame = capture.read()
        
        if ret:
            # Recortar y redimensionar cada frame
            resized_frame = crop_and_resize(frame, individual_width, final_height)
            frames.append(resized_frame)
        else:
            # Reiniciar video si se termina
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(frames) == len(captures):
        # Combinar los frames
        combined_frame = cv2.hconcat(frames)

        # Solo analizar cada 30 frames
        if frame_counter % analyze_interval == 0:
            previous_detections = []  # Limpiar detecciones previas
            row, col, _ = combined_frame.shape
            _max = max(col, row)
            result = np.zeros((_max, _max, 3), np.uint8)
            result[0:row, 0:col] = combined_frame
            
            input_image = result
            blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
            net.setInput(blob)
            predictions = net.forward()

            # Procesar las predicciones
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
                        box = [left, top, width, height]
                        boxes.append(box)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

            for i in indexes:
                previous_detections.append({
                    "box": boxes[i],
                    "class_id": class_ids[i]
                })

        # Dibujar las detecciones (última o actual)
        for detection in previous_detections:
            box = detection["box"]
            class_id = detection["class_id"]
            cv2.rectangle(combined_frame, box, (0, 255, 255), 2)
            cv2.rectangle(combined_frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(combined_frame, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        # Calcular los FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Mostrar los FPS en el frame
        cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar el frame combinado con las detecciones
        cv2.imshow("Combined Video with Detections", combined_frame)

    # Incrementar el contador de frames
    frame_counter += 1

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
for capture in captures:
    capture.release()
cv2.destroyAllWindows()