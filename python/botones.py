
import numpy as np
import cv2
import time
import PySimpleGUI as sg
from threading import Thread

# Configuración global para pausar/reanudar
paused = False

def toggle_pause():
    global paused
    paused = not paused

# Variables para el desplazamiento
shift = 0

def move_left():
    global shift
    shift -= 1

def move_right():
    global shift
    shift += 1

# Cargar el modelo YOLOv5
net = cv2.dnn.readNet('/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx')

# Configuración de las rutas de los videos
video_paths = ["/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4", 
               "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video2.mp4", 
               "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video3.mp4",
               "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video2.mp4",
               "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4"]
captures = [cv2.VideoCapture(path) for path in video_paths]

# Dimensiones de la resolución final
final_width = 1280
final_height = 400
individual_width = 426  # Ancho de cada video
individual_height = 240  # Alto de cada video

# Lista de clases para la detección
class_list = []
with open("/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

# Función para redimensionar y recortar el frame
def crop_and_resize(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height))

# Contador de fotogramas y detecciones previas
frame_counter = 0
analyze_interval = 5
previous_detections = []

# Función para procesar el video
def process_video(window):
    global frame_counter, previous_detections, combined_frame, paused, shift
    while True:
        start_time = time.time()

        frames = []
        for i in range(3):
            capture = captures[(i + shift) % len(captures)]
            ret, frame = capture.read()
            
            if ret:
                resized_frame = crop_and_resize(frame, individual_width, individual_height)
                frames.append(resized_frame)
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if len(frames) == 3:
            combined_frame = cv2.hconcat(frames)
            if not paused:
                if frame_counter % analyze_interval == 0:
                    previous_detections = []
                    row, col, _ = combined_frame.shape
                    _max = max(col, row)
                    result = np.zeros((_max, _max, 3), np.uint8)
                    result[0:row, 0:col] = combined_frame
                    
                    input_image = result
                    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
                    net.setInput(blob)
                    predictions = net.forward()

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

                
                    for detection in previous_detections:
                        box = detection["box"]
                        class_id = detection["class_id"]
                        cv2.rectangle(combined_frame, box, (0, 255, 255), 2)
                        cv2.rectangle(combined_frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
                        cv2.putText(combined_frame, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            imgbytes = cv2.imencode('.png', combined_frame)[1].tobytes()
            window.write_event_value('-IMAGE-', imgbytes)

        frame_counter += 1

        time.sleep(0.03)

# Crear la interfaz gráfica con PySimpleGUI
layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Button('<', key='-LEFT-', size=(3, 1)), sg.Button('>', key='-RIGHT-', size=(3, 1)), sg.Button('Pausar/Reanudar', key='-TOGGLE-', size=(12, 1))]
]

window = sg.Window('Video Detections', layout, size=(1280, 400))

# Iniciar el hilo de procesamiento de video
video_thread = Thread(target=process_video, args=(window,))
video_thread.daemon = True
video_thread.start()

# Bucle principal de la interfaz gráfica
while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break
    elif event == '-LEFT-':
        move_left()
    elif event == '-RIGHT-':
        move_right()
    elif event == '-TOGGLE-':
        toggle_pause()
    elif event == '-IMAGE-':
        window['-IMAGE-'].update(data=values['-IMAGE-'])

window.close()