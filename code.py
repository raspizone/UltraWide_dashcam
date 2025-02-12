import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import PySimpleGUI as sg
from threading import Thread
from queue import Queue

def check_available_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index + 1):
        capture = cv2.VideoCapture(i)
        if capture.isOpened():
            ret, frame = capture.read()
            if ret:
                available_cameras.append(i)
            capture.release()
    return available_cameras

paused = False
captures = []
CAMERA = False
model_path = r"C:\Users\model.tflite"
label_path = r"C:\Users\coco_labels.txt"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)

if CAMERA:
    video_paths = check_available_cameras()
else:
    video_paths = [
        r"C:\Users\video1.mp4",
        r"C:\Users\video2.mp4",
        r"C:\Users\video3.mp4",
        r"C:\Users\video4.mp4"
    ]

def initialize_cameras(video_paths):
    cameras = []
    for path in video_paths:
        capture = cv2.VideoCapture(path)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 15)
            cameras.append(capture)
        else:
            print(f"Error: No se pudo abrir la cámara o archivo de video {path}")
    return cameras

captures = initialize_cameras(video_paths)
n_sources = len(captures)

def move_camera_left(index):
    if index > 0:
        captures[index], captures[index - 1] = captures[index - 1], captures[index]

def move_camera_right(index):
    if index < len(captures) - 1:
        captures[index], captures[index + 1] = captures[index + 1], captures[index]

def process_video(queue, width=1280):
    height = int(720 / n_sources)
    global paused
    while True:
        if not paused:
            frames = []
            for capture in captures:
                ret, frame = capture.read()
                if ret:
                    frame = cv2.resize(frame, (width // len(captures), height))
                    frames.append(frame)
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            if len(frames) == len(captures):
                combined_frame = cv2.hconcat(frames)
                image = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
                left_half = image.crop((0, 0, 640, 180))
                right_half = image.crop((640, 0, 1280, 180))
                new_image = Image.new("RGB", (640, 360))
                new_image.paste(left_half, (0, 0))
                new_image.paste(right_half, (0, 180))
                new_image_np = np.array(new_image)
                final_frame = cv2.cvtColor(new_image_np, cv2.COLOR_RGB2BGR)

                _, scale = common.set_resized_input(
                    interpreter, new_image.size, lambda size: new_image.resize(size, Image.LANCZOS))
                interpreter.invoke()
                
                objs = detect.get_objects(interpreter, 0.4, scale)
                draw = ImageDraw.Draw(new_image)
                for obj in objs:
                    bbox = obj.bbox
                    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
                    draw.text((bbox.xmin, bbox.ymin), f'{labels.get(obj.id, obj.id)} {obj.score:.2f}', fill='red')

                top_half = new_image.crop((0, 0, 640, 180))
                bottom_half = new_image.crop((0, 180, 640, 360))
                final_image = Image.new("RGB", (1280, 180))
                final_image.paste(top_half, (0, 0))
                final_image.paste(bottom_half, (640, 0))
                final_frame = np.array(final_image)
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
                queue.put(final_frame)

layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Button(f'<-{i+1}', key=f'-LEFT-{i}', size=(20, 1)) for i in range(len(captures))],
    [sg.Button(f'->{i+1}', key=f'-RIGHT-{i}', size=(20, 1)) for i in range(len(captures))],
    [sg.Button('Pausar/Reanudar', key='-TOGGLE-', size=(12, 1))]
]

window = sg.Window('Video Detections', layout, finalize=True)
frame_queue = Queue()
video_thread = Thread(target=process_video, args=(frame_queue,))
video_thread.daemon = True
video_thread.start()

while True:
    event, _ = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED:
        break
    elif event.startswith('-LEFT-'):
        index = int(event.split('-')[-1])
        move_camera_left(index)
    elif event.startswith('-RIGHT-'):
        index = int(event.split('-')[-1])
        move_camera_right(index)
    elif event == '-TOGGLE-':
        paused = not paused

    if not frame_queue.empty():
        frame = frame_queue.get()
        _, buffer = cv2.imencode('.png', frame)
        imgbytes = buffer.tobytes()
        window['-IMAGE-'].update(data=imgbytes)

window.close()
