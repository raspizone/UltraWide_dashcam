import cv2

# Función para extraer la ventana de la mitad de un video
def extract_half_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        split_point = w // 2 + 50  # Cortar en una mitad no exacta
        half_frame = frame[:, :split_point]

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (split_point, h))

        out.write(half_frame)

    cap.release()
    if out is not None:
        out.release()


# Ruta del video de entrada y salida
video_path = "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/video1.mp4"
output_path = "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/cap1.mp4",
# Extraer la ventana de la mitad del video
extract_half_video(video_path, output_path)
print(f"El video de la mitad ha sido guardado en {output_path}")