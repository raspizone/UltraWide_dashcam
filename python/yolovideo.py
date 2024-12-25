import numpy as np
import cv2

# Cargar el modelo YOLOv5
net = cv2.dnn.readNet('/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx')

# Configuración de las rutas de los videos
video_paths = ["/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4", "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video2.mp4", "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video3.mp4"]  # Rutas de tus videos
captures = [cv2.VideoCapture(path) for path in video_paths]

# Dimensiones de la resolución final
final_width = 1280
final_height = 400
individual_width = final_width // len(video_paths)

# Lista de clases para la detección
class_list = []
with open("/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# Función para redimensionar y recortar el frame
def crop_and_resize(frame, target_width, target_height):
    h, w, _ = frame.shape
    crop_left = (w - target_width) // 2  # Recorte izquierdo
    crop_right = crop_left + target_width  # Recorte derecho
    crop_top = (h - target_height) // 2  # Recorte superior
    crop_bottom = crop_top + target_height  # Recorte inferior
    cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
    return cropped_frame

# Procesamiento de cada frame de video
while True:
    frames = []
    for capture in captures:
        ret, frame = capture.read()
        
        if ret:
            # Recortar y redimensionar cada frame
            resized_frame = crop_and_resize(frame, individual_width, final_height)
            frames.append(resized_frame)
            print()
        else:
            # Reiniciar video si se termina
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(frames) == len(captures):
        # Combinar los frames
        for i, frame in enumerate(frames):
            cv2.imshow("Combined ", frame)
        
        combined_frame = cv2.hconcat(frames)
        



        # Formatear para YOLOv5
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
                if (classes_scores[class_id] > .25):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        # Dibujar las cajas de detección
        for i in range(len(result_class_ids)):
            box = result_boxes[i]
            class_id = result_class_ids[i]
            cv2.rectangle(combined_frame, box, (0, 255, 255), 2)
            cv2.rectangle(combined_frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(combined_frame, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

        # Mostrar el frame combinado con las detecciones
        cv2.imshow("Combined Video with Detections", combined_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
for capture in captures:
    capture.release()
cv2.destroyAllWindows()