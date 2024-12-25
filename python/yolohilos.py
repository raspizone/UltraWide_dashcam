import numpy as np
import cv2
from multiprocessing import Process, Queue
import time

# Configuración de las rutas de los videos
video_paths = [
    "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4",
    "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video2.mp4",
    "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4"
]
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
    crop_left = (w - target_width) // 2
    crop_right = crop_left + target_width
    crop_top = (h - target_height) // 2
    crop_bottom = crop_top + target_height
    cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
    return cropped_frame

# Función para procesar un frame con YOLOv5
def process_frame_with_yolo(frame, queue, frame_index):
    # Cargar una nueva instancia del modelo YOLOv5
    net = cv2.dnn.readNet('/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx')
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Procesar las predicciones
    class_ids, confidences, boxes = [], [], []
    rows, cols, _ = frame.shape
    x_factor, y_factor = cols / 640, rows / 640

    for row in predictions[0]:
        confidence = row[4]
        if confidence >= 0.4:
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            if class_scores[class_id] > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[:4]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width, height = int(w * x_factor), int(h * y_factor)
                boxes.append([left, top, width, height])

    # Dibujar detecciones en el frame
    for box, class_id in zip(boxes, class_ids):
        cv2.rectangle(frame, box, (0, 255, 0), 2)
        label = f"{class_list[class_id]}: {confidences[class_ids.index(class_id)]:.2f}"
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Añadir el frame procesado a la cola
    queue.put((frame_index, frame))

# Bucle principal
while True:
    frames = []
    for capture in captures:
        ret, frame = capture.read()
        
        if ret:
            resized_frame = crop_and_resize(frame, individual_width, final_height)
            frames.append(resized_frame)
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(frames) == len(captures):
        result_queue = Queue()
        processes = []

        # Crear procesos para procesar cada frame
        for i, frame in enumerate(frames):
            process = Process(target=process_frame_with_yolo, args=(frame, result_queue, i))
            process.start()
            processes.append(process)

        # Esperar a que haya 3 objetos en la cola
        while result_queue.qsize() < 3:
            pass  # Aquí podrías poner un pequeño retraso si lo prefieres

       
        # Recopilar los frames procesados en orden
        processed_frames = [None] * len(frames)
        for _ in range(len(frames)):
            frame_index, processed_frame = result_queue.get()
            processed_frames[frame_index] = processed_frame

     
        if len(processed_frames) ==3:
        # Combinar los frames procesados
            combined_frame = cv2.hconcat(processed_frames)
            cv2.imshow("Combined Video with Detections", combined_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
for capture in captures:
    capture.release()
cv2.destroyAllWindows()
